"""
Feldolgozás státusz ellenőrző
"""

import time
from pathlib import Path


def check_status():
    """Ellenőrzi a feldolgozás státuszát"""
    output_file = Path('output/glint_removal_full_video.mp4')
    
    print("🔄 Glint Removal Videó Feldolgozás - Státusz")
    print("="*60)
    
    if not output_file.exists():
        print("⏳ Feldolgozás még nem indult el vagy nem hozott létre fájlt")
        return
    
    # Fájlméret ellenőrzése
    file_size = output_file.stat().st_size
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"📁 Fájl: {output_file}")
    print(f"💾 Jelenlegi méret: {file_size_mb:.1f} MB")
    
    # Becsült végméret (empirikus: ~800x800@112fps ≈ 5-10 MB/sec)
    # 45649 képkocka @ 111.84 fps = ~408 sec videó
    # Becsült végméret: 2-4 GB
    
    estimated_final_mb = 3000  # konzervatív becslés
    progress_percent = (file_size_mb / estimated_final_mb) * 100
    
    print(f"📊 Becsült haladás: {min(progress_percent, 100):.1f}%")
    print(f"🎯 Várható végméret: ~{estimated_final_mb} MB")
    
    if progress_percent < 100:
        print(f"⏳ Még várakozási idő: ~{(100-progress_percent)/100 * 15:.0f} perc")
    else:
        print("✅ Feldolgozás valószínűleg befejezett!")
    
    print("="*60)
    print("\n💡 Tipp: Nyiss egy új terminált és futtasd újra ezt a scriptet")
    print("   a frissített státuszért!")


if __name__ == "__main__":
    check_status()
