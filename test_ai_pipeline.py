"""
AI Pipeline Teszt - MediaPipe
"""

from ai_pupil_pipeline import AIEyeTrackingPipeline
import yaml

# Konfiguráció módosítása teszteléshez
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

config['video']['end_frame'] = 50
config['output']['show_preview'] = False
config['output']['save_annotated'] = True

with open('config_ai_test.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(config, f)

print("="*60)
print("AI PUPIL DETECTION - MEDIAPIPE TESZT")
print("="*60)
print("Első 50 képkocka MediaPipe Iris-szal\n")

pipeline = AIEyeTrackingPipeline("config_ai_test.yaml")
pipeline.run()

print("\n✅ AI Teszt befejezve!")
print("Eredmények:")
print("  - output/ai_pupil_data.json")
print("  - output/ai_annotated_output.mp4")
