import React, { useState } from 'react'

interface VideoFormProps {
  onJobCreated: (jobId: string) => void
  disabled?: boolean
}

interface FormData {
  brand_name: string
  brand_description: string
  target_audience: string
  tone: string
  duration: number
  call_to_action: string
}

interface LogoUpload {
  file: File | null
  preview: string | null
}

export function VideoForm({ onJobCreated, disabled = false }: VideoFormProps) {
  const [formData, setFormData] = useState<FormData>({
    brand_name: '',
    brand_description: '',
    target_audience: 'general audience',
    tone: 'friendly',
    duration: 30,
    call_to_action: 'Take action now'
  })
  
  const [logo, setLogo] = useState<LogoUpload>({
    file: null,
    preview: null
  })
  
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [error, setError] = useState<string | null>(null)
  
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (!formData.brand_name || !formData.brand_description) {
      setError('Brand name and description are required')
      return
    }
    
    setIsSubmitting(true)
    setError(null)
    
    try {
      // Create FormData for multipart request (supports logo upload)
      const submitData = new FormData()
      submitData.append('brand_name', formData.brand_name)
      submitData.append('brand_description', formData.brand_description)
      
      // Add logo file if selected
      if (logo.file) {
        submitData.append('logo', logo.file)
      }
      
      const response = await fetch('/api/generate', {
        method: 'POST',
        body: submitData // No Content-Type header needed for FormData
      })
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }
      
      const result = await response.json()
      onJobCreated(result.job_id)
      
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Generation failed')
    } finally {
      setIsSubmitting(false)
    }
  }
  
  const handleChange = (field: keyof FormData, value: string | number) => {
    setFormData(prev => ({ ...prev, [field]: value }))
  }
  
  const handleLogoUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    
    // Validate file type
    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file')
      return
    }
    
    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
      setError('Logo file size must be less than 10MB')
      return
    }
    
    // Create preview URL
    const previewUrl = URL.createObjectURL(file)
    setLogo({ file, preview: previewUrl })
    
    // Clear any previous errors
    if (error) setError(null)
  }
  
  const removeLogo = () => {
    if (logo.preview) {
      URL.revokeObjectURL(logo.preview)
    }
    setLogo({ file: null, preview: null })
  }
  
  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Brand Name */}
        <div>
          <label className="block text-sm font-medium text-blue-200 mb-2">
            Brand Name *
          </label>
          <input
            type="text"
            value={formData.brand_name}
            onChange={(e) => handleChange('brand_name', e.target.value)}
            className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-blue-300 focus:outline-none focus:border-blue-400"
            placeholder="Enter your brand name"
            disabled={disabled}
            required
          />
        </div>
        
        {/* Target Audience */}
        <div>
          <label className="block text-sm font-medium text-blue-200 mb-2">
            Target Audience
          </label>
          <select
            value={formData.target_audience}
            onChange={(e) => handleChange('target_audience', e.target.value)}
            className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-blue-400"
            disabled={disabled}
          >
            <option value="general audience">General Audience</option>
            <option value="young adults">Young Adults (18-30)</option>
            <option value="professionals">Professionals</option>
            <option value="parents">Parents</option>
            <option value="seniors">Seniors (55+)</option>
            <option value="entrepreneurs">Entrepreneurs</option>
          </select>
        </div>
      </div>
      
      {/* Brand Description */}
      <div>
        <label className="block text-sm font-medium text-blue-200 mb-2">
          Brand Description *
        </label>
        <textarea
          value={formData.brand_description}
          onChange={(e) => handleChange('brand_description', e.target.value)}
          className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-blue-300 focus:outline-none focus:border-blue-400"
          placeholder="Describe your brand, product, or service..."
          rows={4}
          disabled={disabled}
          required
        />
      </div>
      
      {/* Logo Upload Section */}
      <div>
        <label className="block text-sm font-medium text-blue-200 mb-2">
          Brand Logo (Optional)
        </label>
        <div className="space-y-4">
          {/* File Input */}
          <div className="flex items-center justify-center w-full">
            <label className="flex flex-col items-center justify-center w-full h-32 border-2 border-white/20 border-dashed rounded-lg cursor-pointer bg-white/5 hover:bg-white/10 transition-colors">
              <div className="flex flex-col items-center justify-center pt-5 pb-6">
                <svg className="w-8 h-8 mb-4 text-blue-300" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 16">
                  <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2"/>
                </svg>
                <p className="mb-2 text-sm text-blue-300">
                  <span className="font-semibold">Click to upload logo</span>
                </p>
                <p className="text-xs text-blue-400">PNG, JPG, or SVG (max 10MB)</p>
              </div>
              <input
                type="file"
                className="hidden"
                accept="image/*"
                onChange={handleLogoUpload}
                disabled={disabled}
              />
            </label>
          </div>
          
          {/* Logo Preview */}
          {logo.preview && (
            <div className="relative bg-white/5 border border-white/20 rounded-lg p-4">
              <div className="flex items-center space-x-4">
                <div className="flex-shrink-0">
                  <img
                    src={logo.preview}
                    alt="Logo preview"
                    className="w-16 h-16 object-contain bg-white/10 rounded border border-white/20"
                  />
                </div>
                <div className="flex-grow">
                  <p className="text-sm text-white font-medium">{logo.file?.name}</p>
                  <p className="text-xs text-blue-300">
                    {logo.file ? `${(logo.file.size / (1024 * 1024)).toFixed(2)} MB` : ''}
                  </p>
                </div>
                <button
                  type="button"
                  onClick={removeLogo}
                  disabled={disabled}
                  className="text-red-400 hover:text-red-300 transition-colors"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M6 18L18 6M6 6l12 12"/>
                  </svg>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Tone */}
        <div>
          <label className="block text-sm font-medium text-blue-200 mb-2">
            Tone
          </label>
          <select
            value={formData.tone}
            onChange={(e) => handleChange('tone', e.target.value)}
            className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-blue-400"
            disabled={disabled}
          >
            <option value="friendly">Friendly</option>
            <option value="professional">Professional</option>
            <option value="energetic">Energetic</option>
            <option value="sophisticated">Sophisticated</option>
            <option value="casual">Casual</option>
          </select>
        </div>
        
        {/* Duration */}
        <div>
          <label className="block text-sm font-medium text-blue-200 mb-2">
            Duration (seconds)
          </label>
          <select
            value={formData.duration}
            onChange={(e) => handleChange('duration', parseInt(e.target.value))}
            className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white focus:outline-none focus:border-blue-400"
            disabled={disabled}
          >
            <option value={15}>15 seconds</option>
            <option value={30}>30 seconds</option>
            <option value={45}>45 seconds</option>
            <option value={60}>60 seconds</option>
          </select>
        </div>
        
        {/* Call to Action */}
        <div>
          <label className="block text-sm font-medium text-blue-200 mb-2">
            Call to Action
          </label>
          <input
            type="text"
            value={formData.call_to_action}
            onChange={(e) => handleChange('call_to_action', e.target.value)}
            className="w-full px-4 py-3 bg-white/10 border border-white/20 rounded-lg text-white placeholder-blue-300 focus:outline-none focus:border-blue-400"
            placeholder="Call to action text"
            disabled={disabled}
          />
        </div>
      </div>
      
      {/* Error Message */}
      {error && (
        <div className="bg-red-500/20 border border-red-500/50 rounded-lg p-4">
          <p className="text-red-200">{error}</p>
        </div>
      )}
      
      {/* Submit Button */}
      <button
        type="submit"
        disabled={disabled || isSubmitting}
        className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-bold py-4 px-8 rounded-lg transition-all duration-200 disabled:cursor-not-allowed"
      >
        {isSubmitting ? 'Starting Generation...' : 'Generate Video Ad'}
      </button>
    </form>
  )
}