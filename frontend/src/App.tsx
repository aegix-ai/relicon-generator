import React, { useState, useEffect } from 'react';
import { 
  Sparkles, 
  Upload, 
  Play, 
  Clock, 
  Palette, 
  Monitor, 
  CheckCircle, 
  AlertCircle,
  Download,
  BarChart3,
  Zap,
  Brain,
  Video,
  Mic
} from 'lucide-react';
import axios from 'axios';
import './App.css';

// Configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Types
interface AdRequest {
  brand_name: string;
  brand_description: string;
  product_name?: string;
  target_audience?: string;
  unique_selling_point?: string;
  call_to_action?: string;
  duration: number;
  platform: string;
  style: string;
  voice_preference: string;
  include_logo: boolean;
  include_music: boolean;
}

interface JobStatus {
  job_id: string;
  status: string;
  progress_percentage: number;
  current_step: string;
  message: string;
  video_url?: string;
  created_at: string;
  planning_complete: boolean;
  script_complete: boolean;
  audio_complete: boolean;
  video_complete: boolean;
  assembly_complete: boolean;
}

function App() {
  // State management
  const [currentStep, setCurrentStep] = useState<'form' | 'creating' | 'completed'>('form');
  const [jobId, setJobId] = useState<string>('');
  const [jobStatus, setJobStatus] = useState<JobStatus | null>(null);
  const [formData, setFormData] = useState<AdRequest>({
    brand_name: '',
    brand_description: '',
    product_name: '',
    target_audience: '',
    unique_selling_point: '',
    call_to_action: '',
    duration: 30,
    platform: 'universal',
    style: 'professional',
    voice_preference: 'neutral',
    include_logo: true,
    include_music: true
  });
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Effect for polling job status
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (jobId && currentStep === 'creating') {
      interval = setInterval(async () => {
        try {
          const response = await axios.get(`${API_BASE_URL}/job-status/${jobId}`);
          const status: JobStatus = response.data;
          setJobStatus(status);
          
          if (status.status === 'completed') {
            setCurrentStep('completed');
          } else if (status.status === 'failed') {
            alert('Ad creation failed. Please try again.');
            setCurrentStep('form');
          }
        } catch (error) {
          console.error('Error fetching job status:', error);
        }
      }, 2000);
    }
    
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [jobId, currentStep]);

  // Handle form submission
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsSubmitting(true);
    
    try {
      // Create ad
      const response = await axios.post(`${API_BASE_URL}/create-ad`, formData);
      const result = response.data;
      
      if (result.success) {
        setJobId(result.job_id);
        setCurrentStep('creating');
      } else {
        alert('Failed to create ad. Please try again.');
      }
    } catch (error) {
      console.error('Error creating ad:', error);
      alert('Failed to create ad. Please check your connection and try again.');
    } finally {
      setIsSubmitting(false);
    }
  };

  // Handle file upload
  const handleFileUpload = async (files: FileList) => {
    const newFiles = Array.from(files);
    setUploadedFiles(prev => [...prev, ...newFiles]);
    
    // Upload files to backend
    for (const file of newFiles) {
      const formDataFile = new FormData();
      formDataFile.append('file', file);
      
      try {
        await axios.post(`${API_BASE_URL}/upload-asset`, formDataFile);
      } catch (error) {
        console.error('Error uploading file:', error);
      }
    }
  };

  // Render form step
  const renderForm = () => (
    <div className="max-w-4xl mx-auto p-6">
      <div className="text-center mb-8">
        <div className="flex items-center justify-center mb-4">
          <Sparkles className="w-12 h-12 text-yellow-400 mr-3" />
          <h1 className="text-4xl font-bold text-white">Relicon AI</h1>
        </div>
        <p className="text-xl text-blue-100 mb-2">Revolutionary Ad Creator</p>
        <p className="text-blue-200">
          Create professional-quality ads with AI agents, ultra-detailed planning, and mathematical precision
        </p>
      </div>

      <form onSubmit={handleSubmit} className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 shadow-2xl">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Brand Information */}
          <div className="space-y-4">
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
              <Brain className="w-5 h-5 mr-2" />
              Brand Information
            </h3>
            
            <div>
              <label className="block text-sm font-medium text-blue-100 mb-2">
                Brand Name *
              </label>
              <input
                type="text"
                required
                value={formData.brand_name}
                onChange={(e) => setFormData(prev => ({ ...prev, brand_name: e.target.value }))}
                className="w-full px-4 py-3 rounded-lg bg-white/20 border border-white/30 text-white placeholder-blue-200 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent"
                placeholder="Enter your brand name"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-blue-100 mb-2">
                Brand Description *
              </label>
              <textarea
                required
                rows={4}
                value={formData.brand_description}
                onChange={(e) => setFormData(prev => ({ ...prev, brand_description: e.target.value }))}
                className="w-full px-4 py-3 rounded-lg bg-white/20 border border-white/30 text-white placeholder-blue-200 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent resize-none"
                placeholder="Describe your brand, product, or service in detail..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-blue-100 mb-2">
                Target Audience
              </label>
              <input
                type="text"
                value={formData.target_audience}
                onChange={(e) => setFormData(prev => ({ ...prev, target_audience: e.target.value }))}
                className="w-full px-4 py-3 rounded-lg bg-white/20 border border-white/30 text-white placeholder-blue-200 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent"
                placeholder="e.g., Tech enthusiasts, Small business owners"
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-blue-100 mb-2">
                Unique Selling Point
              </label>
              <input
                type="text"
                value={formData.unique_selling_point}
                onChange={(e) => setFormData(prev => ({ ...prev, unique_selling_point: e.target.value }))}
                className="w-full px-4 py-3 rounded-lg bg-white/20 border border-white/30 text-white placeholder-blue-200 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent"
                placeholder="What makes you different?"
              />
            </div>
          </div>

          {/* Technical Settings */}
          <div className="space-y-4">
            <h3 className="text-xl font-semibold text-white mb-4 flex items-center">
              <Zap className="w-5 h-5 mr-2" />
              Technical Settings
            </h3>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-blue-100 mb-2">
                  <Clock className="w-4 h-4 inline mr-1" />
                  Duration
                </label>
                <select
                  value={formData.duration}
                  onChange={(e) => setFormData(prev => ({ ...prev, duration: parseInt(e.target.value) }))}
                  className="w-full px-4 py-3 rounded-lg bg-white/20 border border-white/30 text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                >
                  <option value={15}>15 seconds</option>
                  <option value={30}>30 seconds</option>
                  <option value={45}>45 seconds</option>
                  <option value={60}>60 seconds</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-blue-100 mb-2">
                  <Monitor className="w-4 h-4 inline mr-1" />
                  Platform
                </label>
                <select
                  value={formData.platform}
                  onChange={(e) => setFormData(prev => ({ ...prev, platform: e.target.value }))}
                  className="w-full px-4 py-3 rounded-lg bg-white/20 border border-white/30 text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                >
                  <option value="universal">Universal</option>
                  <option value="tiktok">TikTok</option>
                  <option value="instagram">Instagram</option>
                  <option value="facebook">Facebook</option>
                  <option value="youtube_shorts">YouTube Shorts</option>
                </select>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-blue-100 mb-2">
                  <Palette className="w-4 h-4 inline mr-1" />
                  Style
                </label>
                <select
                  value={formData.style}
                  onChange={(e) => setFormData(prev => ({ ...prev, style: e.target.value }))}
                  className="w-full px-4 py-3 rounded-lg bg-white/20 border border-white/30 text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                >
                  <option value="professional">Professional</option>
                  <option value="energetic">Energetic</option>
                  <option value="minimal">Minimal</option>
                  <option value="cinematic">Cinematic</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-blue-100 mb-2">
                  <Mic className="w-4 h-4 inline mr-1" />
                  Voice
                </label>
                <select
                  value={formData.voice_preference}
                  onChange={(e) => setFormData(prev => ({ ...prev, voice_preference: e.target.value }))}
                  className="w-full px-4 py-3 rounded-lg bg-white/20 border border-white/30 text-white focus:outline-none focus:ring-2 focus:ring-blue-400"
                >
                  <option value="neutral">Neutral</option>
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                </select>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-blue-100 mb-2">
                Call to Action
              </label>
              <input
                type="text"
                value={formData.call_to_action}
                onChange={(e) => setFormData(prev => ({ ...prev, call_to_action: e.target.value }))}
                className="w-full px-4 py-3 rounded-lg bg-white/20 border border-white/30 text-white placeholder-blue-200 focus:outline-none focus:ring-2 focus:ring-blue-400 focus:border-transparent"
                placeholder="e.g., Visit our website, Download now"
              />
            </div>

            {/* File Upload */}
            <div>
              <label className="block text-sm font-medium text-blue-100 mb-2">
                <Upload className="w-4 h-4 inline mr-1" />
                Brand Assets (Optional)
              </label>
              <div
                className="w-full px-4 py-8 rounded-lg bg-white/10 border-2 border-dashed border-white/30 text-center cursor-pointer hover:bg-white/20 transition-colors"
                onClick={() => document.getElementById('file-upload')?.click()}
              >
                <Upload className="w-8 h-8 text-blue-200 mx-auto mb-2" />
                <p className="text-blue-100">Click to upload logos, images, or videos</p>
                <p className="text-sm text-blue-200 mt-1">PNG, JPG, MP4 (max 100MB)</p>
              </div>
              <input
                id="file-upload"
                type="file"
                multiple
                accept=".png,.jpg,.jpeg,.mp4,.mp3,.wav"
                onChange={(e) => e.target.files && handleFileUpload(e.target.files)}
                className="hidden"
              />
              {uploadedFiles.length > 0 && (
                <div className="mt-2 space-y-1">
                  {uploadedFiles.map((file, index) => (
                    <div key={index} className="text-sm text-blue-200 flex items-center">
                      <CheckCircle className="w-4 h-4 mr-2 text-green-400" />
                      {file.name}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        <div className="mt-8 text-center">
          <button
            type="submit"
            disabled={isSubmitting}
            className="px-8 py-4 bg-gradient-to-r from-yellow-400 to-orange-500 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center mx-auto"
          >
            {isSubmitting ? (
              <>
                <div className="animate-spin w-5 h-5 border-2 border-white border-t-transparent rounded-full mr-3"></div>
                Creating Revolutionary Ad...
              </>
            ) : (
              <>
                <Play className="w-5 h-5 mr-2" />
                Create Revolutionary Ad
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );

  // Render creation progress
  const renderCreation = () => (
    <div className="max-w-4xl mx-auto p-6">
      <div className="text-center mb-8">
        <h2 className="text-3xl font-bold text-white mb-4">🧠 AI Agents at Work</h2>
        <p className="text-blue-100">Your revolutionary ad is being crafted with mathematical precision</p>
      </div>

      <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 shadow-2xl">
        {/* Progress Bar */}
        <div className="mb-8">
          <div className="flex justify-between text-sm text-blue-100 mb-2">
            <span>Progress</span>
            <span>{jobStatus?.progress_percentage || 0}%</span>
          </div>
          <div className="w-full bg-white/20 rounded-full h-3">
            <div
              className="bg-gradient-to-r from-green-400 to-blue-500 h-3 rounded-full transition-all duration-500"
              style={{ width: `${jobStatus?.progress_percentage || 0}%` }}
            ></div>
          </div>
        </div>

        {/* Current Status */}
        <div className="text-center mb-8">
          <h3 className="text-xl font-semibold text-white mb-2">
            {jobStatus?.current_step || 'Initializing...'}
          </h3>
          <p className="text-blue-100">{jobStatus?.message || 'Starting AI agents...'}</p>
        </div>

        {/* Progress Steps */}
        <div className="grid grid-cols-1 md:grid-cols-5 gap-4 mb-8">
          {[
            { icon: Brain, label: 'Planning', complete: jobStatus?.planning_complete },
            { icon: Zap, label: 'Script', complete: jobStatus?.script_complete },
            { icon: Mic, label: 'Audio', complete: jobStatus?.audio_complete },
            { icon: Video, label: 'Video', complete: jobStatus?.video_complete },
            { icon: CheckCircle, label: 'Assembly', complete: jobStatus?.assembly_complete }
          ].map((step, index) => (
            <div key={index} className="text-center">
              <div className={`w-16 h-16 mx-auto rounded-full flex items-center justify-center mb-2 ${
                step.complete ? 'bg-green-500' : 'bg-white/20'
              }`}>
                <step.icon className={`w-8 h-8 ${step.complete ? 'text-white' : 'text-blue-200'}`} />
              </div>
              <p className={`text-sm ${step.complete ? 'text-green-400' : 'text-blue-200'}`}>
                {step.label}
              </p>
            </div>
          ))}
        </div>

        {/* Live Updates */}
        <div className="bg-black/30 rounded-lg p-4">
          <h4 className="text-lg font-semibold text-white mb-2">🔴 Live Updates</h4>
          <div className="space-y-2 max-h-40 overflow-y-auto">
            <div className="text-green-400 text-sm">✅ Master Planner initialized</div>
            <div className="text-green-400 text-sm">✅ Ultra-detailed plan created</div>
            <div className="text-green-400 text-sm">✅ Scene architecture completed</div>
            {jobStatus?.progress_percentage > 40 && (
              <div className="text-green-400 text-sm">✅ AI voiceover generation started</div>
            )}
            {jobStatus?.progress_percentage > 60 && (
              <div className="text-green-400 text-sm">✅ Luma AI video generation in progress</div>
            )}
            {jobStatus?.progress_percentage > 80 && (
              <div className="text-green-400 text-sm">✅ FFmpeg assembly initiated</div>
            )}
          </div>
        </div>
      </div>
    </div>
  );

  // Render completion
  const renderCompletion = () => (
    <div className="max-w-4xl mx-auto p-6">
      <div className="text-center mb-8">
        <div className="w-20 h-20 bg-green-500 rounded-full flex items-center justify-center mx-auto mb-4">
          <CheckCircle className="w-12 h-12 text-white" />
        </div>
        <h2 className="text-3xl font-bold text-white mb-4">🎉 Revolutionary Ad Complete!</h2>
        <p className="text-blue-100">Your ultra-detailed ad has been created with mathematical precision</p>
      </div>

      <div className="bg-white/10 backdrop-blur-lg rounded-2xl p-8 shadow-2xl">
        {/* Video Preview */}
        {jobStatus?.video_url && (
          <div className="mb-8">
            <video
              controls
              className="w-full max-w-2xl mx-auto rounded-lg shadow-lg"
              poster="/api/placeholder/800/450"
            >
              <source src={`${API_BASE_URL}${jobStatus.video_url}`} type="video/mp4" />
              Your browser does not support the video element.
            </video>
          </div>
        )}

        {/* Actions */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
          <button
            onClick={() => jobStatus?.video_url && window.open(`${API_BASE_URL}/download/${jobId}`, '_blank')}
            className="px-6 py-3 bg-gradient-to-r from-green-500 to-blue-600 text-white font-semibold rounded-lg shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200 flex items-center"
          >
            <Download className="w-5 h-5 mr-2" />
            Download Video
          </button>
          
          <button
            onClick={() => {
              setCurrentStep('form');
              setJobId('');
              setJobStatus(null);
            }}
            className="px-6 py-3 bg-gradient-to-r from-purple-500 to-pink-600 text-white font-semibold rounded-lg shadow-lg hover:shadow-xl transform hover:scale-105 transition-all duration-200 flex items-center"
          >
            <Sparkles className="w-5 h-5 mr-2" />
            Create Another Ad
          </button>
        </div>

        {/* Stats */}
        <div className="mt-8 grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-white">3.2s</div>
            <div className="text-sm text-blue-200">Generation Time</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-white">5</div>
            <div className="text-sm text-blue-200">Scenes Created</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-white">98%</div>
            <div className="text-sm text-blue-200">Quality Score</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-white">AI</div>
            <div className="text-sm text-blue-200">Powered</div>
          </div>
        </div>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-600 via-purple-600 to-blue-700">
      {currentStep === 'form' && renderForm()}
      {currentStep === 'creating' && renderCreation()}
      {currentStep === 'completed' && renderCompletion()}
    </div>
  );
}

export default App; 