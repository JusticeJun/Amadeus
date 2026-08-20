# Amadeus

ESP32-S3-N16R8 기반 320×480 세로형 탁상 AI 비서 프로젝트입니다.

현재 단계는 승인된 기본 캐릭터 이미지를 ST7796 LCD에 표시하는 것입니다. 마이크와 서보는 초기화하지 않습니다.

## 명령

PlatformIO Core가 PATH에 없다면 다음 실행 파일을 사용합니다.

```powershell
& "$env:USERPROFILE\.platformio\penv\Scripts\pio.exe" run
& "$env:USERPROFILE\.platformio\penv\Scripts\pio.exe" device list
```

업로드 포트는 자동 감지 결과를 확인한 뒤 명시합니다.

## 이미지 변환 및 업로드

원본 PNG는 `assets/source/characters/`에 보존합니다. 파생 RGB565 파일만 `data/`에 생성합니다.

```powershell
& "$env:USERPROFILE\.platformio\penv\Scripts\python.exe" tools/convert_rgb565.py assets/source/characters/neutral_default_320x480.png data/neutral_default.rgb565
& "$env:USERPROFILE\.platformio\penv\Scripts\pio.exe" run --target uploadfs
& "$env:USERPROFILE\.platformio\penv\Scripts\pio.exe" run --target upload
```

`uploadfs`는 캐릭터 데이터, `upload`는 펌웨어를 각각 전송하는 서로 다른 작업입니다.

## 기본 화면 효과

디스플레이가 켜진 동안 약한 CRT 스캔라인과 가장자리 감광을 유지합니다. 간헐적인 작은 가로 찢김과 드문 넓은 수평 동기 밀림 효과도 기본 활성화됩니다. 효과 강도와 확률은 `include/ProjectConfig.h`에서 조절합니다.
