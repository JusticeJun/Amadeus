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

## Windows PC control tool

`pc_control` converts supported requests into validated `PcAction` values before
calling the Windows adapter. App launches are restricted to Chrome, Notepad,
Calculator, and VS Code, use resolved executable paths, and always run with
`shell=False`. The first release also supports volume changes in 10-point steps,
absolute volume from 0 to 100, mute/unmute, and best-effort system media keys for
play/pause, next, and previous.

The parser is behind a `PcActionParser` protocol so action extraction can be
replaced without coupling it to semantic capability routing or OS calls. Tests
use injected fake controllers and do not change the workstation.

Independent requests such as `오늘 날씨 알려주고 크롬 켜줘` may execute both
tools. Conditional requests such as `비 오면 크롬 켜줘` retain both capabilities
in `RouteDecision`, but are marked `planning_required` and execute nothing until
a dependency-aware planner is implemented.

## Apple Music PWA tool

`music_control` separates minimal rule-based action extraction from the
production `MusicAction` and `AppleMusicPwaController` layers. Its backend uses
only the dedicated Amadeus Chrome profile through a localhost CDP endpoint; it
does not inspect or copy the user's normal Chrome profile.

Song and playlist candidates must match normalized names and available
title/artist metadata uniquely. The controller verifies now-playing metadata
after side effects, caches playlist discovery, and may return a marked partial
or stale playlist snapshot when a full refresh is unavailable. Existing generic
OS media actions are retained for compatibility, while music utterances prefer
`music_control` to avoid duplicate playback side effects.

Music interpretation uses deterministic transport and status commands as a fast
path. Entity-heavy, contextual, or conversational requests may use a strict
structured semantic LLM fallback. That component can only produce bounded
`MusicActionSequence` values and search alternatives; Apple Music catalog and
playlist IDs, candidate selection, playback, and success remain controlled and
verified by the deterministic resolver/controller layers. Provider failure or
invalid structured output does not authorize a side effect.
