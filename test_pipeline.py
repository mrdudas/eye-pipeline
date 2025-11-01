"""
Pipeline tesztelése - első néhány képkockán próba
"""

from pupil_pipeline import EyeTrackingPipeline
import yaml

# Konfiguráció betöltése és módosítása teszteléshez
with open('config.yaml', 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Csak az első 50 képkockát dolgozzuk fel teszteléshez
config['video']['end_frame'] = 50
config['output']['show_preview'] = False  # Ne nyissa meg az ablakot
config['output']['save_annotated'] = True

# Teszt konfiguráció mentése
with open('config_test.yaml', 'w', encoding='utf-8') as f:
    yaml.dump(config, f, default_flow_style=False)

print("Pipeline tesztelés - első 50 képkocka")
print("=" * 50)

# Pipeline futtatása
pipeline = EyeTrackingPipeline("config_test.yaml")
pipeline.run()

print("\nTeszt befejezve!")
print(f"Eredmények: output/pupil_data.json")
print(f"Videó: output/annotated_output.mp4")
