import React, { useState } from 'react';
import './App.css';

function App() {
  const [selectedBodyPart, setSelectedBodyPart] = useState('');
  const [image, setImage] = useState(null);
  const [output, setOutput] = useState('');
  const [showWarning, setShowWarning] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      setImage(file);
      setShowWarning(false);
      setError('');
    }
  };

  const handlePredict = async () => {
    if (image && selectedBodyPart) {
      setLoading(true);
      setError('');
      setOutput('');
      setShowWarning(false);
      
      try {
        // Create form data for sending to backend
        const formData = new FormData();
        formData.append('image', image);
        formData.append('bodyPart', selectedBodyPart);
        
        // Send request to Flask backend
        const response = await fetch('http://localhost:5000/api/predict', {
          method: 'POST',
          body: formData,
        });
        
        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.error || 'Something went wrong');
        }
        
        const data = await response.json();
        
        // Update output based on prediction
        setOutput(`${data.body_part.toUpperCase()}: ${data.prediction} (Confidence: ${data.confidence})`);
      } catch (error) {
        setError(error.message || 'An error occurred during prediction');
      } finally {
        setLoading(false);
      }
    } else {
      setOutput('');
      setShowWarning(true);
    }
  };

  return (
    <div className="App" style={{
      minHeight: '100vh',
      background: 'linear-gradient(to right, #e0f7fa, #e1f5fe)',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      padding: '40px',
      fontFamily: 'Segoe UI, sans-serif',
    }}>
      <h1 style={{ color: '#2c3e50', marginBottom: '40px' }}>
        Bone Abnormalities Detection Using Machine Learning
      </h1>

      {/* Button Row */}
      <div style={{
        display: 'flex',
        gap: '30px',
        marginBottom: '30px',
        flexWrap: 'wrap',
        justifyContent: 'center'
      }}>
        {/* Upload Button */}
        <label style={{
          backgroundColor: '#5dade2',
          color: 'white',
          padding: '12px 24px',
          borderRadius: '8px',
          cursor: 'pointer'
        }}>
          Upload Image
          <input type="file" accept="image/*" onChange={handleImageUpload} style={{ display: 'none' }} />
        </label>

        {/* Dropdown */}
        <select
          value={selectedBodyPart}
          onChange={(e) => setSelectedBodyPart(e.target.value)}
          style={{
            padding: '12px 20px',
            borderRadius: '8px',
            border: '1px solid #ccc',
            fontSize: '16px'
          }}
        >
          <option value="">Select Body Part</option>
          <option value="shoulder">Shoulder</option>
          <option value="hand">Hand</option>
          <option value="forearm">Forearm</option>
          <option value="wrist">Wrist</option>
          <option value="humerus">Humerus</option>
          <option value="elbow">Elbow</option>
          <option value="finger">Finger</option>
        </select>
      </div>

      {/* Image preview */}
      {image && (
        <div style={{ marginBottom: '20px' }}>
          <img 
            src={URL.createObjectURL(image)} 
            alt="Preview" 
            style={{ 
              maxHeight: '200px', 
              maxWidth: '100%', 
              borderRadius: '8px',
              boxShadow: '0 4px 8px rgba(0,0,0,0.1)'
            }} 
          />
        </div>
      )}

      {/* Predict Button */}
      <button
        onClick={handlePredict}
        disabled={loading}
        style={{
          padding: '14px 32px',
          fontSize: '16px',
          backgroundColor: loading ? '#95a5a6' : '#28b463',
          color: 'white',
          border: 'none',
          borderRadius: '10px',
          cursor: loading ? 'not-allowed' : 'pointer',
          marginBottom: '20px'
        }}
      >
        {loading ? 'Processing...' : 'Predict Output'}
      </button>

      {/* Warning or Output */}
      {showWarning && (
        <div style={{
          padding: '15px 20px',
          backgroundColor: '#f8d7da',
          color: '#721c24',
          borderLeft: '6px solid #f44336',
          borderRadius: '8px',
          fontSize: '16px',
          maxWidth: '80%',
          textAlign: 'center'
        }}>
          Please upload an image and select a body part.
        </div>
      )}

      {error && (
        <div style={{
          padding: '15px 20px',
          backgroundColor: '#f8d7da',
          color: '#721c24',
          borderLeft: '6px solid #f44336',
          borderRadius: '8px',
          fontSize: '16px',
          maxWidth: '80%',
          textAlign: 'center'
        }}>
          Error: {error}
        </div>
      )}

      {output && (
        <div style={{
          padding: '15px 20px',
          backgroundColor: '#d5f5e3',
          color: '#145a32',
          borderLeft: '6px solid #1e8449',
          borderRadius: '8px',
          fontSize: '16px',
          maxWidth: '80%',
          textAlign: 'center'
        }}>
          {output}
        </div>
      )}
    </div>
  );
}

export default App;