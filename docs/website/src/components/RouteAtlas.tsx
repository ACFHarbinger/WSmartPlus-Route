import { useState } from "react";

const stops = [
  {
    id: "D",
    label: "Depot",
    x: 11,
    y: 56,
    detail: "The route begins and returns to a constrained origin.",
  },
  {
    id: "1",
    label: "North sector",
    x: 31,
    y: 27,
    detail:
      "Candidate stop selected by a policy under capacity and length limits.",
  },
  {
    id: "2",
    label: "Transfer",
    x: 50,
    y: 63,
    detail:
      "A high-fill stop where the local route trade-off becomes visible.",
  },
  {
    id: "3",
    label: "East sector",
    x: 71,
    y: 34,
    detail:
      "A profitable detour balanced against distance and future demand.",
  },
  {
    id: "4",
    label: "Return",
    x: 89,
    y: 70,
    detail: "The route closes while preserving feasibility.",
  },
];

export default function RouteAtlas() {
  const [selected, setSelected] = useState(0);
  const route = stops.map((stop) => `${stop.x},${stop.y}`).join(" ");

  return (
    <div className="route-atlas" aria-label="Interactive example route atlas">
      <svg
        viewBox="0 0 100 100"
        role="img"
        aria-label="Illustrative route connecting five stops"
      >
        <defs>
          <pattern
            id="atlas-grid"
            width="8"
            height="8"
            patternUnits="userSpaceOnUse"
          >
            <path
              d="M 8 0 L 0 0 0 8"
              fill="none"
              stroke="#ffffff"
              strokeOpacity=".06"
              strokeWidth=".2"
            />
          </pattern>
          <linearGradient id="atlas-route" x1="0" x2="1">
            <stop offset="0" stopColor="#3dffa8" />
            <stop offset="1" stopColor="#4cc9f0" />
          </linearGradient>
        </defs>
        <rect width="100" height="100" fill="url(#atlas-grid)" rx="2" />
        <polyline
          points={route}
          fill="none"
          stroke="#62e7e0"
          strokeOpacity=".2"
          strokeWidth="3"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        <polyline
          points={route}
          fill="none"
          stroke="url(#atlas-route)"
          strokeWidth=".7"
          strokeDasharray="1.7 1.2"
          strokeLinejoin="round"
          strokeLinecap="round"
        />
        {stops.map((stop, index) => (
          <g
            key={stop.id}
            className="atlas-stop"
            onClick={() => setSelected(index)}
            onKeyDown={(event) => {
              if (event.key === "Enter" || event.key === " ") {
                event.preventDefault();
                setSelected(index);
              }
            }}
            role="button"
            tabIndex={0}
            aria-label={`${stop.label}: ${stop.detail}`}
            aria-pressed={selected === index}
          >
            {selected === index && (
              <circle
                cx={stop.x}
                cy={stop.y}
                r="5"
                fill="none"
                stroke="#f0b429"
                strokeOpacity=".55"
                strokeWidth=".35"
              />
            )}
            <circle
              cx={stop.x}
              cy={stop.y}
              r={selected === index ? "2.4" : "1.7"}
              fill={selected === index ? "#f0b429" : "#62e7e0"}
            />
            <text
              x={stop.x}
              y={stop.y - 4}
              textAnchor="middle"
              fill="#cbd4e4"
              fontSize="2.6"
              fontFamily="DM Mono, monospace"
            >
              {stop.id}
            </text>
          </g>
        ))}
      </svg>
      <div className="atlas-caption">
        <span className="eyebrow">
          Selected stop · {stops[selected].label}
        </span>
        <p>{stops[selected].detail}</p>
      </div>
    </div>
  );
}
