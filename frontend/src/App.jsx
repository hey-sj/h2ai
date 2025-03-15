// src/App.jsx
import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'

function App() {
  const [message, setMessage] = useState('')
  const [loading, setLoading] = useState(true)
  const [userInput, setUserInput] = useState('')
  const [responseMessage, setResponseMessage] = useState('')

  useEffect(() => {
    // Fetch data from Flask backend
    axios.get('/api/hello')
      .then(response => {
        setMessage(response.data.message)
        setLoading(false)
      })
      .catch(error => {
        console.error("Error fetching data:", error)
        setLoading(false)
      })
  }, [])

  const handleSubmit = async (e) => {
    e.preventDefault()
    try {
      const response = await axios.post('/api/submit', {text: userInput})
      setResponseMessage(response.data.response)
      setUserInput('') // Clear the input after submission
    } catch (error) {
      console.error("Error submitting data:", error)
      setResponseMessage('Error submitting data')
    }
  }
  return (
    <div className="App">
      <header className="App-header">
        <h1>Flask + React App with Vite</h1>
        {loading ? (
          <p>Loading message from backend...</p>
        ) : (
          <p>Message from backend: {message}</p>
        )}


        <div className ='input-section' style={{ marginTop: '2rem' }}>
          <h2>Send a Message to the Backend</h2>
          <form onSubmit={handleSubmit}>
          <input
              type="text"
              value={userInput}
              onChange={(e) => setUserInput(e.target.value)}
              placeholder="Enter your message"
              style={{
                padding: '0.5rem',
                marginRight: '0.5rem',
                borderRadius: '4px',
                border: '1px solid #ccc',
                width: '300px'
              }}
            />
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
              Send
            </button>
          </form>

          {responseMessage && (
            <div style={{marginTop: '1rem'}}>
              <h3>Response:</h3> 
              <p>{responseMessage}</p>
            </div>
          )}
        </div> 
      </header>
    </div>
  )
}

export default App