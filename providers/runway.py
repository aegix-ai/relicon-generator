"""
Fully working Runway AI video generation provider.

Key features:
- Correct API host & headers (api.dev.runwayml.com + X-Runway-Version).
- Image→Video and Text→Video (Text→Image -> Image→Video chain).
- Proper payload keys: promptText, promptImage, model, ratio, duration.
- Ratios mapped per model family (Gen-4 vs Gen-3 Alpha Turbo).
- Robust polling with jitter & backoff; 429/5xx handling.
- Task cancel, download helper, cost estimates.
"""

import os
import time
import json
import random
import math
import logging
from typing import Dict, Any, Optional, List, Tuple, Union

import requests

try:
    # Keep your existing interface if present
    from interfaces.video_generator import VideoGenerator  # type: ignore
except Exception:  # pragma: no cover
    class VideoGenerator:  # minimal fallback interface
        pass


RunwayPromptImage = Union[str, List[Dict[str, str]]]


class RunwayProvider(VideoGenerator):
    """
    Runway AI video generation service implementation (Gen-4 / Gen-3 Turbo).
    Docs:
      - Using the API (endpoints, headers, cURL): https://docs.dev.runwayml.com/guides/using-the-api
      - Models (ratios, pricing): https://docs.dev.runwayml.com/guides/models/
      - Versioning & 2024-11-06 changes (ratio, promptImage): https://docs.dev.runwayml.com/api-details/versions/2024-11-06
      - SDK/endpoint mapping & tasks: https://docs.dev.runwayml.com/api-details/sdks
    """

    API_BASE = "https://api.dev.runwayml.com/v1"
    API_VERSION = "2024-11-06"  # required header
    DEFAULT_MODEL = "gen4_turbo"  # good default per docs
    # Pricing: credits -> USD (1 credit = $0.01)
    PRICING_CREDITS_PER_SEC = {
        "gen4_turbo": 5,
        "gen3a_turbo": 5,
        # add others if you enable them:
        "gen4_aleph": 15,
    }
    CREDIT_USD = 0.01

    # Allowed output ratios per model (per docs)
    RATIOS_GEN4 = {"1280:720", "720:1280", "1104:832", "832:1104", "960:960", "1584:672"}
    RATIOS_GEN3 = {"1280:768", "768:1280"}

    def __init__(self, *, api_key_env: Tuple[str, ...] = ("RUNWAY_API_KEY", "RUNWAYML_API_SECRET")):
        self.api_key = None
        for name in api_key_env:
            self.api_key = os.environ.get(name)
            if self.api_key:
                break
        if not self.api_key:
            raise ValueError("Set RUNWAY_API_KEY (or RUNWAYML_API_SECRET) in env")

        self.session = requests.Session()
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "X-Runway-Version": self.API_VERSION,
        }
        self.log = logging.getLogger(self.__class__.__name__)

    # ---------------- Public API (matches your previous class) ----------------

    def text_to_video(self, prompt: str, aspect_ratio: str = "9:16", **kwargs) -> str:
        """Text-only → video by chaining text_to_image then image_to_video."""
        return self.generate_video(prompt, aspect_ratio=aspect_ratio, image_url=None, **kwargs)

    def image_to_video(self, prompt: str, image_url: str, aspect_ratio: str = "9:16", **kwargs) -> str:
        """Image + text → video."""
        return self.generate_video(prompt, aspect_ratio=aspect_ratio, image_url=image_url, **kwargs)

    def generate_video(
        self,
        prompt: str,
        aspect_ratio: str = "9:16",
        image_url: Optional[str] = None,
        *,
        model: str = DEFAULT_MODEL,
        duration: int = 5,
        seed: Optional[int] = None,
        watermark: Optional[bool] = None,
        force_unique: bool = False,
        prompt_image_position: str = "first",  # 'first' or 'last' when passing list
        reference_images: Optional[List[Dict[str, str]]] = None,  # for text->image advanced use
    ) -> str:
        """
        Returns a direct (ephemeral) URL to the generated MP4. Download it ASAP.

        - If image_url is provided → POST /v1/image_to_video
        - Else → POST /v1/text_to_image to get an image, then /v1/image_to_video
        """
        model = (model or self.DEFAULT_MODEL).strip()
        ratio = self._aspect_to_ratio(aspect_ratio, model)
        duration = int(max(1, min(duration, 10)))  # 5s/10s typical per docs
        prompt_text = self._enhance_prompt(prompt) if force_unique else prompt.strip()

        if not self.validate_prompt(prompt_text):
            raise ValueError("Prompt rejected by local validator (possible policy violation or too long).")

        # Step 1: possibly make an image
        prompt_image: RunwayPromptImage
        if image_url:
            prompt_image = self._format_prompt_image(image_url, prompt_image_position)
        else:
            # Text -> Image
            tti_payload = {
                "model": "gen4_image",
                "promptText": prompt_text,
                # Pick a matching orientation for nicer framing
                "ratio": "1080:1920" if self._is_portrait_ratio(ratio) else "1920:1080",
            }
            if reference_images:
                # reference_images: List[{"uri": str, "tag": str}]
                tti_payload["referenceImages"] = reference_images
            tti_task = self._post("/text_to_image", tti_payload)
            tti_done = self._wait_for_task(tti_task["id"])
            out_imgs = tti_done.get("output") or []
            if not out_imgs:
                raise ValueError("Text-to-image produced no output")
            prompt_image = out_imgs[0]

        # Step 2: Image -> Video
        i2v_payload: Dict[str, Any] = {
            "model": model,
            "promptImage": prompt_image,
            "promptText": prompt_text,
            "ratio": ratio,
            "duration": duration,
        }
        if seed is not None:
            i2v_payload["seed"] = int(seed)
        if watermark is not None:
            i2v_payload["watermark"] = bool(watermark)

        i2v_task = self._post("/image_to_video", i2v_payload)
        done = self._wait_for_task(i2v_task["id"])
        out = done.get("output") or []
        if not out:
            raise ValueError("Image-to-video produced no output")
        return out[0]

    def download_video(self, video_url: str, output_path: str, max_retries: int = 3) -> bool:
        """Download the (temporary) output URL to disk."""
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        for attempt in range(max_retries):
            try:
                with self.session.get(video_url, stream=True, timeout=60) as r:
                    r.raise_for_status()
                    with open(output_path, "wb") as f:
                        for chunk in r.iter_content(8192):
                            if chunk:
                                f.write(chunk)
                return os.path.exists(output_path) and os.path.getsize(output_path) > 0
            except Exception as e:
                if attempt == max_retries - 1:
                    raise ValueError(f"Download failed after {max_retries} attempts: {e}")
                time.sleep(2 ** attempt + random.random())
        return False

    def get_generation_cost(self, duration: int = 5, model: str = DEFAULT_MODEL) -> float:
        """Rough $ estimate from docs (credits per second; 1 credit = $0.01)."""
        cps = self.PRICING_CREDITS_PER_SEC.get(model, 5)
        return round(max(1, min(int(duration), 10)) * cps * self.CREDIT_USD, 2)

    def validate_prompt(self, prompt: str) -> bool:
        """Simple local validation before hitting the API."""
        if not prompt or len(prompt) > 500:
            return False
        banned = ["nsfw", "explicit", "illegal", "harmful"]
        return not any(t in prompt.lower() for t in banned)

    # ---------------- Extra helpers you might want ----------------

    def get_task(self, task_id: str) -> Dict[str, Any]:
        """Fetch raw task JSON; useful for debugging."""
        return self._request("GET", f"/tasks/{task_id}")

    def cancel_task(self, task_id: str) -> None:
        """Cancel or delete a task (204 No Content when successful)."""
        self._request("DELETE", f"/tasks/{task_id}", expect_json=False)

    # ---------------- Internals ----------------

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", path, json=payload)

    def _request(self, method: str, path: str, *, json: dict | None = None, expect_json: bool = True) -> dict:
        url = f"{self.API_BASE}{path}"
        attempt = 0
        while True:
            try:
                resp = self.session.request(method, url, headers=self.headers, json=json, timeout=30)
                if resp.status_code == 401:
                    raise ValueError("Unauthorized (401): bad/missing API key or version header")
                if resp.status_code == 400:
                    # ← show the actual problem from Runway instead of retrying
                    try:
                        detail = resp.json()
                    except Exception:
                        detail = {"error": resp.text}
                    raise ValueError(f"Bad Request (400) calling {path}: {detail}")

                if resp.status_code in (429, 500, 502, 503, 504):
                    attempt += 1
                    if attempt > 5:
                        resp.raise_for_status()
                    retry_after = resp.headers.get("Retry-After")
                    delay = int(retry_after) if retry_after and retry_after.isdigit() else min(30, 2 ** attempt)
                    time.sleep(delay)
                    continue

                resp.raise_for_status()
                return {} if not expect_json else resp.json()
            except requests.RequestException as e:
                attempt += 1
                if attempt > 5:
                    raise ValueError(f"Runway API request failed: {e}") from e
                time.sleep(min(30, 2 ** attempt))
    def _wait_for_task(self, task_id: str, timeout_s: int = 600) -> Dict[str, Any]:
        """
        Poll tasks.retrieve until SUCCEEDED / FAILED / CANCELED.
        Per docs, poll ≥5s and add jitter/backoff to be polite under load.
        """
        start = time.time()
        delay = 5.0
        while True:
            task = self.get_task(task_id)
            status = (task.get("status") or "").upper()
            if status == "SUCCEEDED":
                return task
            if status in ("FAILED", "CANCELED"):
                # Bubble up failure code/reason when present
                code = task.get("failure_code") or task.get("code")
                reason = task.get("failure_reason") or task.get("message")
                raise ValueError(f"Task {status.lower()}: {code or ''} {reason or ''}".strip())

            if time.time() - start > timeout_s:
                raise TimeoutError(f"Task polling timed out after {timeout_s}s")

            # exponential-ish backoff with jitter, cap ~15s
            sleep_for = min(15.0, delay) + random.uniform(0, 0.75)
            time.sleep(sleep_for)
            delay *= 1.15

    @staticmethod
    def _enhance_prompt(prompt: str) -> str:
        add = (
            "professional commercial cinematography, dynamic movement, premium lighting, "
            "cinematic depth of field, smooth camera motion, modern color grading"
        )
        return f"{prompt.strip()}. {add}"

    @staticmethod
    def _is_portrait_ratio(ratio: str) -> bool:
        w, h = (int(x) for x in ratio.split(":"))
        return h > w

    def _aspect_to_ratio(self, aspect: str, model: str) -> str:
        """
        Convert friendly aspect ratios (e.g., '9:16') to API 'W:H' tokens
        that depend on the model family.
        """
        aspect = (aspect or "").strip()
        # Gen-4 allowed outputs (720p family + square + alt wides)
        gen4_map = {
            "16:9": "1280:720",
            "9:16": "720:1280",
            "1:1": "960:960",
            "4:3": "1104:832",
            "3:4": "832:1104",
            "21:9": "1584:672",  # wide alt in docs
        }
        # Gen-3 Turbo allowed outputs
        gen3_map = {
            "16:9": "1280:768",
            "9:16": "768:1280",
        }

        if model.startswith("gen4"):
            ratio = gen4_map.get(aspect, gen4_map["9:16"])
            if ratio not in self.RATIOS_GEN4:
                ratio = "720:1280"
            return ratio

        # default to Gen-3
        ratio = gen3_map.get(aspect, gen3_map["9:16"])
        if ratio not in self.RATIOS_GEN3:
            ratio = "768:1280"
        return ratio

    @staticmethod
    def _format_prompt_image(image: str, position: str = "first") -> RunwayPromptImage:
        """
        Accept either a single URL/data URI string or wrap it in [{uri, position}]
        to control whether the image starts or ends the clip (supported by 2024-11-06).
        """
        pos = "last" if str(position).lower().strip() == "last" else "first"
        # You can pass a string directly, but wrapping allows 'last' positioning.
        return [{"uri": image, "position": pos}]


# ---------------- Example usage ----------------
if __name__ == "__main__":
    """
    Scene 3 example:
      - Text→Video (auto text→image, portrait 9:16, Gen-4 Turbo, 5s)
      - Then download immediately (URLs are ephemeral).
    """
    logging.basicConfig(level=logging.INFO)
    provider = RunwayProvider()

    prompt = "Calm seaside at golden hour; gentle dolly-in toward a lighthouse"
    try:
        url = provider.text_to_video(prompt, aspect_ratio="9:16", duration=5, force_unique=True)
        print("Video URL:", url)
        provider.download_video(url, "./out/scene3.mp4")
        print("Saved to ./out/scene3.mp4")
    except Exception as e:
        print("Generation failed:", e)
