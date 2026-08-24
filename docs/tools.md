# PC Bridge tools

Tools provide facts or action results to the conversation pipeline. They do not
write Chris's final reply. A `SemanticRouter` selects the external capabilities
required by the request, and `ToolExecutor` runs the registered tools. Each tool
returns a compact `ToolResult` and builds capability-specific LLM context. Groq
turns the collected context into the final reply and emotion before TTS and
ESP32 state output.

```text
user input -> SemanticRouter -> RouteDecision -> ToolExecutor
           -> Tool results -> Groq -> emotion -> TTS -> ESP32
```

Routing backends live under `pc_bridge/app/routing/`; external capability
implementations live under `pc_bridge/app/tools/`. Register both in
`pc_bridge/app/cli.py`. `RouteDecision` supports more than one capability, while
the current rule backend remains a local fast path. Confidence fallback and
dependency-aware planning are intentionally separate future layers rather than
responsibilities of tools.

Keep credentials in environment variables, Tool results compact, and failures
isolated from the main conversation. The evaluation corpus is a holdout asset
for comparing routing backends and must not be used as classifier training data.

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
