// src/App.jsx
import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [loading, setLoading] = useState(false)
  const [userInput, setUserInput] = useState('')
  const [currentStep, setCurrentStep] = useState('input') // 'input', 'processing', or 'question'
  const [question, setQuestion] = useState('')
  const [finalResult, setFinalResult] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!userInput.trim()) return
    
    setLoading(true)
    setCurrentStep('processing')
    
    try {
      const response = await axios.post('/api/process-text', { text: userInput })
      setQuestion(response.data.question)
      setCurrentStep('question')
    } catch (error) {
      console.error("Error processing text:", error)
      setCurrentStep('input')
    } finally {
      setLoading(false)
    }
  }

  const handleAnswer = async (answer) => {
    setLoading(true)
    
    try {
      const response = await axios.post('/api/answer', { 
        text: userInput,
        question: question,
        answer: answer 
      })
      
      setFinalResult(response.data.result)
      setCurrentStep('result')
    } catch (error) {
      console.error("Error submitting answer:", error)
    } finally {
      setLoading(false)
    }
  }

  const resetForm = () => {
    setUserInput('')
    setQuestion('')
    setFinalResult('')
    setCurrentStep('input')
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>Text Analysis App</h1>
        
        {currentStep === 'input' && (
          <div className="input-section">
            <h2>Enter Your Text</h2>
            <form onSubmit={handleSubmit}>
              <textarea
                value={userInput}
                onChange={(e) => setUserInput(e.target.value)}
                placeholder="Enter your text here..."
                style={{
                  padding: '0.5rem',
                  borderRadius: '4px',
                  border: '1px solid #ccc',
                  width: '400px',
                  height: '150px',
                  marginBottom: '1rem'
                }}
              />
              <div>
                <button 
                  type="submit"
                  style={{
                    padding: '0.5rem 1rem',
                    backgroundColor: '#4CAF50',
                    color: 'white',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer'
                  }}
                >
                  Submit
                </button>
              </div>
            </form>
          </div>
        )}
        
        {currentStep === 'processing' && (
          <div className="processing-section">
            <h2>Processing Your Text...</h2>
            <p>Please wait while we analyze your input.</p>
          </div>
        )}
        
        {currentStep === 'question' && (
          <div className="question-section">
            <h2>Question:</h2>
            <p style={{
              fontSize: '1.2rem',
              marginBottom: '2rem',
              maxWidth: '600px'
            }}>
              {question}
            </p>
            
            <div style={{ display: 'flex', gap: '1rem' }}>
              <button
                onClick={() => handleAnswer('yes')}
                style={{
                  padding: '0.75rem 2rem',
                  backgroundColor: '#4CAF50',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '1rem'
                }}
              >
                Yes
              </button>
              
              <button
                onClick={() => handleAnswer('no')}
                style={{
                  padding: '0.75rem 2rem',
                  backgroundColor: '#f44336',
                  color: 'white',
                  border: 'none',
                  borderRadius: '4px',
                  cursor: 'pointer',
                  fontSize: '1rem'
                }}
              >
                No
              </button>
            </div>
          </div>
        )}
        
        {currentStep === 'result' && (
          <div className="result-section">
            <h2>Result:</h2>
            <p style={{
              fontSize: '1.2rem',
              marginBottom: '2rem',
              maxWidth: '600px'
            }}>
              {finalResult}
            </p>
            
            <button
              onClick={resetForm}
              style={{
                padding: '0.75rem 2rem',
                backgroundColor: '#2196F3',
                color: 'white',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                fontSize: '1rem'
              }}
            >
              Start Over
            </button>
          </div>
        )}
        
        {loading && (
          <div className="loading-overlay" style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            backgroundColor: 'rgba(0,0,0,0.5)',
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            zIndex: 1000
          }}>
            <div style={{
              backgroundColor: 'white',
              padding: '2rem',
              borderRadius: '8px',
              color: 'black'
            }}>
              Processing...
            </div>
          </div>
        )}
      </header>
    </div>
  )
}

export default App