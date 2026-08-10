export default function NetworkField() {
  const nodes = [[8, 20], [21, 62], [35, 35], [49, 78], [57, 18], [68, 54], [81, 29], [94, 70]];
  const edges = [[0, 1], [0, 2], [1, 3], [2, 3], [2, 4], [3, 5], [4, 6], [5, 7], [6, 7]];
  return (
    <svg className="network-field" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
      <defs>
        <linearGradient id="route-glow" x1="0" x2="1">
          <stop offset="0" stopColor="#62e7e0" stopOpacity="0" />
          <stop offset=".5" stopColor="#62e7e0" stopOpacity=".72" />
          <stop offset="1" stopColor="#f0b55b" stopOpacity="0" />
        </linearGradient>
        <filter id="node-blur"><feGaussianBlur stdDeviation="1" /></filter>
      </defs>
      {edges.map(([a, b]) => <line key={`${a}-${b}`} x1={nodes[a][0]} y1={nodes[a][1]} x2={nodes[b][0]} y2={nodes[b][1]} stroke="url(#route-glow)" strokeWidth=".18" />)}
      {nodes.map(([x, y], index) => <g key={`${x}-${y}`}>
        <circle cx={x} cy={y} r={index % 3 === 0 ? 2.6 : 1.5} fill="#62e7e0" opacity=".2" filter="url(#node-blur)" />
        <circle cx={x} cy={y} r={index % 3 === 0 ? .8 : .45} fill={index % 3 === 0 ? "#b7fff2" : "#9aa7c0"} />
      </g>)}
    </svg>
  );
}
