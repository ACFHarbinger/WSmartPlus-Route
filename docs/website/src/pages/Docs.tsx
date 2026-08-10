import { ArrowLeft, ArrowUpRight, BookOpen } from "lucide-react";
import { Link } from "react-router-dom";
import { docs, githubRepo } from "../data/site";

export default function Docs() {
  return (
    <div className="interior">
      <Link className="back-link" to="/">
        <ArrowLeft size={15} /> Home
      </Link>
      <header className="page-hero">
        <span className="eyebrow">
          <span className="eyebrow-dot" />
          Documentation · field notes
        </span>
        <h1>
          Start with the map.
          <br />
          <em className="inline-em-route">Then read the detail.</em>
        </h1>
        <p>
          The documentation is the source of truth for architecture,
          configuration, experiments, and the Studio workflow. Links open the
          canonical Markdown in the repository.
        </p>
      </header>

      <div className="doc-list">
        {docs.map((doc) => (
          <a
            key={doc.path}
            className="doc-item"
            href={`${githubRepo}/blob/main/${doc.path}`}
            target="_blank"
            rel="noreferrer"
          >
            <span className="doc-kind">{doc.kind}</span>
            <div className="doc-body">
              <strong>
                <BookOpen size={16} /> {doc.title}
              </strong>
              <span>{doc.blurb}</span>
            </div>
            <ArrowUpRight size={15} className="doc-arrow" />
          </a>
        ))}
      </div>
    </div>
  );
}
