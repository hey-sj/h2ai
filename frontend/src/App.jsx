import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [loading, setLoading] = useState(false)
  const [userInput, setUserInput] = useState('')
  const [currentStep, setCurrentStep] = useState('input') // 'input', 'processing', 'question', 'midresult', 'diagnosis', or 'result'
  const [question, setQuestion] = useState('')
  const [diagnosis, setDiagnosis] = useState('')
  const [finalResult, setFinalResult] = useState('')
  const [midResult, setMidResult] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!userInput.trim()) return
    
    setLoading(true)
    setCurrentStep('processing')
    
    try {
      const response = await axios.post('/api/process-text', { text: userInput })
      setQuestion(response.data.question)
      setCurrentStep(response.data.currentStep)
    } catch (error) {
      console.error("Error processing text:", error)
      setCurrentStep('input')
    } finally {
      setLoading(false)
    }
  }

  const startSecondRound = async () => {
    setLoading(true)
    
    try {
      // This starts the second diagnostic round
      const response = await axios.post('/api/start-second-round', { text: userInput })
      setQuestion(response.data.question)
      setCurrentStep(response.data.currentStep)
    } catch (error) {
      console.error("Error starting second round:", error)
      setCurrentStep('midresult') // Stay at midresult if there's an error
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
        answer: answer,
      })
      
      if (response.data.currentStep === 'midresult') {
        // If we're moving to midresult, save the preliminary diagnosis
        setMidResult(response.data.result)
      } else {
        // Otherwise, continue with normal question flow
        setQuestion(response.data.question)
      }
      
      setCurrentStep(response.data.currentStep)
      
    } catch (error) {
      console.error("Error submitting answer:", error)
    } finally {
      setLoading(false)
    }
  }

  const handleDiagnosis = async (answer) => {
    setLoading(true)
    
    try {
      const response = await axios.post('/api/diagnose', { 
        text: userInput,
        question: question,
        answer: answer,
      })
      
      if (response.data.currentStep === 'result') {
        // If we're moving to final result
        setFinalResult(response.data.result)
      } else {
        // Continue with diagnosis questions
        setQuestion(response.data.question)
        setDiagnosis(response.data.result)
      }
      
      setCurrentStep(response.data.currentStep)
      
    } catch (error) {
      console.error("Error submitting diagnosis answer:", error)
    } finally {
      setLoading(false)
    }
  }

  const resetForm = () => {
    setUserInput('')
    setQuestion('')
    setMidResult('')
    setFinalResult('')
    setCurrentStep('input')
  }

  return (
    <div className="App">
      <header className="App-header">
        <h1>WellTracker</h1>
        
        {currentStep === 'input' && (
          <div className="input-section">
            <h2>Enter Your Symptoms</h2>
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
            
            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
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

        {currentStep === 'midresult' && (
          <div className="result-section">
            <h2>Preliminary Diagnosis:</h2>
            <p style={{
              fontSize: '1.2rem',
              marginBottom: '1rem',
              maxWidth: '600px'
            }}>
              Based on your symptoms, here are the potential diagnoses:
            </p>
            <div style={{
              padding: '1rem',
              borderRadius: '4px',
              marginBottom: '2rem',
              maxWidth: '600px',
              fontSize: '1.5rem',
            }}>
              {midResult}
            </div>
            
            <p style={{
              fontSize: '1rem',
              marginBottom: '1.5rem',
              maxWidth: '600px'
            }}>
              To refine this diagnosis, we can ask more specific questions.
            </p>
            
            <button
              onClick={startSecondRound}
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
              Continue Diagnosis
            </button>
          </div>
        )}
        
        {currentStep === 'diagnosis' && (
          <div className="question-section">
            <h2>Additional Question:</h2>
            <p style={{
              fontSize: '1.2rem',
              marginBottom: '2rem',
              maxWidth: '600px'
            }}>
              {question}
            </p>
            
            <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center' }}>
              <button
                onClick={() => handleDiagnosis('yes')}
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
                onClick={() => handleDiagnosis('no')}
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
            <h2>Final Diagnosis:</h2>
            <p style={{
              fontSize: '1.2rem',
              marginBottom: '1rem',
              maxWidth: '600px'
            }}>
              Based on all your symptoms and responses, here are the most likely diagnoses:
            </p>
            <div style={{
              padding: '1rem',
              borderRadius: '4px',
              marginBottom: '2rem',
              maxWidth: '600px',
              fontSize: '1.5rem',
            }}>
              {finalResult}
            </div>
            
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