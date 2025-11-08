#!/bin/bash

echo "🎯 Predicta API - Render.com Startup"
echo "===================================="

MODEL_FILE="/app/models/weighted_model.pkl"

# FORCE DELETE corrupted model (KeyError: 118 fix)
if [ -f "$MODEL_FILE" ]; then
    echo "🗑️  Deleting old model (force rebuild due to pickle compatibility)"
    rm -f "$MODEL_FILE"
fi

# Always train fresh model on Render
if [ -f "/app/data/merged_all.csv" ]; then
    echo "📊 Training fresh model with Python 3.11 compatible protocol..."
    python app/weighted_trainer.py || {
        echo "⚠️  Training failed, API will use fallback predictor"
    }
else
    echo "⚠️  No dataset found, API will use fallback predictor"
fi

echo ""
echo "🌐 Starting API server on Render.com..."
echo "===================================="
python main_weighted.py
