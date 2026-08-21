# PC Bridge tools

Tools provide facts or action results to the conversation pipeline. They do not
write Chris's final reply. `ToolRouter` selects a matching tool, the tool returns
a compact `ToolResult`, and Groq turns that context into the final reply and
emotion before TTS and ESP32 state output.

```text
user input -> ToolRouter -> optional ToolResult -> Groq -> emotion -> TTS -> ESP32
```

Add future capabilities under `pc_bridge/app/tools/` and register them in
`pc_bridge/app/cli.py`. Keep routing narrow, credentials in environment
variables, responses compact, and failures isolated from the main conversation.

## KMA weather tool

The first implementation uses the Korean Public Data Portal's **KMA Short-term
Forecast Inquiry Service** (`VilageFcstInfoService_2.0`). Current questions use
`getUltraSrtNcst`; today, tonight, and tomorrow questions use `getVilageFcst`.
The tool converts WGS84 latitude/longitude into KMA `nx`/`ny`, tries the latest
safe release and older fallback releases, and sends only summarized facts to
Groq.

Required private configuration in `pc_bridge/.env`:

```dotenv
KMA_SERVICE_KEY=your_data_go_kr_service_key
AMADEUS_LOCATION_NAME=Busan
AMADEUS_LATITUDE=35.1796
AMADEUS_LONGITUDE=129.0756
WEATHER_TIMEOUT_SECONDS=8
```

Both encoded and decoded service keys issued by data.go.kr are accepted. Do not
commit `pc_bridge/.env`.
