import {
  ArrowLeft,
  ArrowUpRight,
  Monitor,
  Play,
  Workflow,
} from "lucide-react";
import { Link } from "react-router-dom";

export default function Studio() {
  return (
    <div className="interior">
      <Link className="back-link" to="/">
        <ArrowLeft size={15} /> Home
      </Link>
      <header className="page-hero">
        <span className="eyebrow">
          <span className="eyebrow-dot" />
          WSmart-Route Studio · desktop
        </span>
        <h1>
          From experiment
          <br />
          <em className="inline-em-route">to operation.</em>
        </h1>
        <p>
          A Tauri desktop workspace for launching, monitoring, comparing, and
          reporting on routing workflows — training dashboards, deck.gl route
          views, and comparative analytics.
        </p>
      </header>

      <div className="card-grid interior-grid">
        <article className="card eco">
          <div className="card-top">
            <span>MONITOR</span>
            <Monitor size={18} />
          </div>
          <h3>Watch work happen</h3>
          <p>
            Follow training, simulation, and process runs without blocking the
            interface.
          </p>
        </article>
        <article className="card route">
          <div className="card-top">
            <span>LAUNCH</span>
            <Play size={18} />
          </div>
          <h3>Configure the next run</h3>
          <p>
            Move from a scenario or model choice to a reproducible command and
            execution record.
          </p>
        </article>
        <article className="card amber">
          <div className="card-top">
            <span>ANALYZE</span>
            <Workflow size={18} />
          </div>
          <h3>Turn outputs into insight</h3>
          <p>
            Explore maps, metrics, comparisons, reports, and generated
            presentation material.
          </p>
        </article>
      </div>

      <section className="text-panel">
        <div className="kicker">Local-first tooling</div>
        <h2>
          The public site explains.
          <br />
          <em className="inline-em-eco">The Studio operates.</em>
        </h2>
        <p>
          Keep the research narrative accessible while the full data and process
          controls stay close to the developer&apos;s machine. Launch with{" "}
          <code className="inline-code">just studio</code> from the repository
          root.
        </p>
        <Link className="btn btn-secondary" to="/docs">
          Read the Studio documentation <ArrowUpRight size={15} />
        </Link>
      </section>
    </div>
  );
}
