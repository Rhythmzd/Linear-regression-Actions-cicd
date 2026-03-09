import pandas as pd
import joblib
import os

def test_model():
    """Test the trained linear regression model"""
    
    # Check if model file exists
    if not os.path.exists('linear_model.pkl'):
        print("❌ Model file not found!")
        return False
    
    # Load the model
    try:
        model = joblib.load('linear_model.pkl')
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False
    
    # Load test data
    try:
        data = pd.read_csv('training_data.csv')
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
        print("✅ Test data loaded successfully")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return False
    
    # Test predictions
    try:
        predictions = model.predict(X)
        print(f"✅ Model predictions generated")
        print(f"📊 Sample predictions: {predictions[:5]}")
        
        # Calculate R-squared
        from sklearn.metrics import r2_score
        r2 = r2_score(y, predictions)
        print(f"📈 R-squared score: {r2:.4f}")
        
        if r2 > 0.5:
            print("✅ Model performance is acceptable")
            return True
        else:
            print("⚠️ Model performance needs improvement")
            return False
            
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Linear Regression Model...")
    success = test_model()
    if success:
        print("🎉 All tests passed!")
    else:
        print("💥 Tests failed!")
        exit(1)
