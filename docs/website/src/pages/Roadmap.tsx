import { ArrowLeft, CircleDot } from "lucide-react";
import { Link } from "react-router-dom";
import { roadmapItems } from "../data/site";

export default function Roadmap() {
  return (
    <div className="interior">
      <Link className="back-link" to="/">
        <ArrowLeft size={15} /> Home
      </Link>
      <header className="page-hero">
        <span className="eyebrow">
          <span className="eyebrow-dot" />
          Roadmap · direction
        </span>
        <h1>
          A direction,
          <br />
          <em className="inline-em-route">not a promise.</em>
        </h1>
        <p>
          The roadmap is a living research boundary. Priorities can change as
          evidence changes. See the full plan in{" "}
          <code className="inline-code">docs/FEATURE_ROADMAP.md</code>.
        </p>
      </header>

      <div className="roadmap-list">
        {roadmapItems.map((item) => (
          <article key={item.phase}>
            <CircleDot size={16} className="roadmap-dot" />
            <span className="eyebrow">{item.phase}</span>
            <h2>{item.title}</h2>
            <p>{item.body}</p>
          </article>
        ))}
      </div>
    </div>
  );
}
