#!/bin/bash
set -e  # Exit on error

echo "🎯 Predicta API - Render.com Startup"
echo "===================================="

# Check Python version
python --version

# Model path
MODEL_FILE="models/weighted_model.pkl"

echo ""
echo "📦 Model Status Check..."
if [ -f "$MODEL_FILE" ]; then
    echo "✅ Model file exists: $MODEL_FILE"
    
    # Test if model is loadable
    python -c "import joblib; joblib.load('$MODEL_FILE'); print('✅ Model is valid')" 2>&1
    
    if [ $? -ne 0 ]; then
        echo "❌ Model is corrupted, deleting..."
        rm -f "$MODEL_FILE"
    fi
fi

# Train model if not exists or corrupted
if [ ! -f "$MODEL_FILE" ]; then
    echo ""
    echo "🔄 No valid model found, training..."
    
    if [ -f "data/merged_all.csv" ]; then
        echo "📊 Dataset found, starting training..."
        python app/weighted_trainer.py
        
        if [ $? -eq 0 ]; then
            echo "✅ Model training completed"
        else
            echo "⚠️  Training failed, will use fallback predictor"
        fi
    else
        echo "⚠️  Dataset not found at data/merged_all.csv"
        echo "📁 Available files:"
        ls -la data/ || echo "data/ directory not found"
    fi
else
    echo "✅ Using existing valid model"
fi

echo ""
echo "🌐 Starting API server..."
echo "===================================="
exec python main_weighted.py
