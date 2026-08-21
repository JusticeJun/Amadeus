# Amadeus

Amadeus는 ESP32-S3와 3.5인치 세로형 LCD로 만드는 탁상형 AI 비서 프로젝트입니다. 차분한 여성 캐릭터 **크리스**를 화면에 표시하고, 음성 대화와 감정 표현, 본체 움직임을 단계적으로 더해 작은 CRT 텔레비전 같은 AI 동반자로 완성하는 것이 목표입니다.

캐릭터 애니메이션은 수백 장의 프레임 대신 감정별 대표 이미지 한 장을 사용합니다. 이미지 전환과 화면의 생동감은 코드로 생성하는 CRT 노이즈와 글리치 효과로 표현합니다.

## 현재 구현 상태

- ESP32-S3-WROOM-1-N16R8의 16MB Flash와 8MB Octal PSRAM 확인
- ST7796 320×480 LCD 세로 화면 출력
- 승인된 크리스 기본 이미지를 RGB565로 변환해 LittleFS에서 표시
- 얇은 CRT 스캔라인과 가장자리 감광을 기본 화면 효과로 적용
- 간헐적인 작은 가로 찢김 글리치
- 드물게 나타나는 넓은 수평 동기 밀림 효과
- 원본 이미지와 ESP32용 파생 에셋을 분리해 보존

현재 LCD 이미지와 CRT 효과는 실제 하드웨어에서 방향, 크기, 색상과 동작을 확인했습니다. 마이크, 서보, 네트워크와 LLM 기능은 아직 초기화하거나 구현하지 않았습니다.

## 하드웨어

- ESP32-S3 DevKitC-1
- ESP32-S3-WROOM-1-N16R8
- 16MB Flash / 8MB Octal PSRAM
- ST7796 SPI TFT LCD, 320×480, 세로 고정
- 개발 및 시리얼 통신: CH343P UART, 115200 baud

향후 INMP441 마이크에는 GPIO4~6, MG996R 서보 2개에는 GPIO16과 GPIO17을 사용할 예정입니다. 이 핀들은 현재 예약되어 있으며 펌웨어에서 초기화하지 않습니다.

## 프로젝트 방향

다음 기능을 순차적으로 개발합니다.

1. CRT 화면 효과 안정화
2. INMP441 음성 입력과 녹음
3. Wi-Fi, STT와 간단한 LLM 대화
4. TTS와 감정 상태 결정
5. 감정별 이미지, 아이콘과 말풍선
6. 외부 전원을 사용하는 서보 2개 제어
7. 유휴 시간 기반 수면 상태

하드웨어 연결과 기능 추가는 한 단계씩 실제 보드에서 검증한 뒤 진행합니다.

## PC AI 브리지

키보드 입력, mock/Groq 대화, GPT-SoVITS 한국어 음성 생성과 USB 시리얼 감정 전송은 `pc_bridge/`에서 실행합니다. 설치, mock 실행, Groq 키 설정, 음성 기준 파일 변환 및 GPT-SoVITS 사용법은 [PC 브리지 안내](pc_bridge/README.md)를 참고하세요.

## 저장소 구조

```text
assets/source/characters/  승인된 원본 캐릭터 이미지
data/                      LittleFS에 업로드할 ESP32용 에셋
include/                   핀 및 프로젝트 설정
lib/                       디스플레이, 에셋, CRT 효과 모듈
src/                       펌웨어 진입점
docs/                      배선, 하드웨어, 로드맵, 문제 해결 문서
tools/                     이미지 변환 도구
```

## 빌드와 업로드

PlatformIO Core가 PATH에 없다면 설치된 실행 파일의 전체 경로를 사용합니다.

```powershell
& "$env:USERPROFILE\.platformio\penv\Scripts\pio.exe" run
& "$env:USERPROFILE\.platformio\penv\Scripts\pio.exe" device list
```

원본 PNG는 변경하지 않습니다. 파생 RGB565 파일을 만든 뒤 파일시스템과 펌웨어를 각각 업로드합니다.

```powershell
& "$env:USERPROFILE\.platformio\penv\Scripts\python.exe" tools/convert_rgb565.py assets/characters/neutral_320x480.png data/neutral.rgb565
& "$env:USERPROFILE\.platformio\penv\Scripts\pio.exe" run --target uploadfs
& "$env:USERPROFILE\.platformio\penv\Scripts\pio.exe" run --target upload
```

`uploadfs`는 캐릭터 데이터를, `upload`는 펌웨어를 전송하는 별도의 작업입니다. CRT 효과의 강도와 발생 확률은 `include/ProjectConfig.h`에서 조절합니다.
