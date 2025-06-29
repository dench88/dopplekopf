# python - << 'EOF'
import io

# 1) Read raw bytes, drop nulls:
raw = open("evaluate_ppo_old.py","rb").read().replace(b"\x00",b"")

# 2) Decode from CP1252 (so those ÔÇ£ sequences become real “quotes”), then re-encode as UTF-8:
text = raw.decode("cp1252")
open("evaluate_ppo_old_fixed.py","w", encoding="utf8").write(text)

print("Written cleaned file → ai_old_version_fixed.py")
# EOF
