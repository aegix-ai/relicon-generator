"use client"

import { useState } from 'react'
import { VideoForm } from '../components/VideoForm'
import JobTracker from '../components/JobTracker'

export default function Home() {
  const [result, setResult] = useState<any>(null)
  const [currentJobId, setCurrentJobId] = useState<string | null>(null)

  const handleJobCreated = (jobId: string) => {
    setCurrentJobId(jobId)
    setResult({ job_id: jobId, status: 'queued' })
    
    // Poll for status updates
    const pollStatus = setInterval(async () => {
      try {
        const statusResponse = await fetch(`http://localhost:8000/api/status/${jobId}`)
        const statusData = await statusResponse.json()
        setResult(statusData)
        
        if (statusData.status === 'completed' || statusData.status === 'failed') {
          clearInterval(pollStatus)
        }
      } catch (error) {
        console.error('Error polling status:', error)
        clearInterval(pollStatus)
      }
    }, 2000)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-900 to-purple-900 p-8">
      <div className="max-w-4xl mx-auto">
        <header className="text-center mb-12">
          <h1 className="text-5xl font-bold text-white mb-4">
            Relicon Clean System
          </h1>
          <p className="text-xl text-blue-200">
            AI-Powered Video Generation - Clean Architecture
          </p>
          <div className="text-sm text-green-400 mt-2 font-mono">
            ✓ Running from relicon/ directory
          </div>
        </header>

        <div className="bg-white/10 backdrop-blur-md rounded-lg p-8 shadow-xl">
          <VideoForm 
            onJobCreated={handleJobCreated}
            disabled={!!currentJobId && result?.status === 'processing'}
          />

          {result && (
            <div className="mt-8">
              <JobTracker jobId={result.job_id} />
            </div>
          )}
        </div>

        <div className="mt-8 text-center text-white/60 text-sm">
          <p>Powered by Clean Relicon-Rewrite Architecture</p>
          <p>Cost: $2.42-4.80 per video | Duration: 30 seconds | 3 scenes</p>
        </div>
      </div>
    </div>
  )
}