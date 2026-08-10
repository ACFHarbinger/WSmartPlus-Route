import {
  ArrowLeft,
  ArrowUpRight,
  BarChart3,
  CircleAlert,
  Gauge,
} from "lucide-react";
import { Link } from "react-router-dom";

export default function Benchmarks() {
  return (
    <div className="interior">
      <Link className="back-link" to="/">
        <ArrowLeft size={15} /> Home
      </Link>
      <header className="page-hero">
        <span className="eyebrow">
          <span className="eyebrow-dot" />
          Benchmarks · evidence
        </span>
        <h1>
          Measure the route,
          <br />
          <em className="inline-em-route">not the story.</em>
        </h1>
        <p>
          Benchmarking is where neural policies, exact solvers, heuristics, and
          simulations meet. Results need a problem definition, a budget, and a
          clear comparison.
        </p>
      </header>

      <div className="card-grid interior-grid">
        <article className="card route">
          <div className="card-top">
            <span>COMPARE</span>
            <BarChart3 size={18} />
          </div>
          <h3>Policy versus policy</h3>
          <p>
            Compare quality, cost, distance, overflow, and runtime across common
            instances and scenarios.
          </p>
        </article>
        <article className="card eco">
          <div className="card-top">
            <span>TELEMETRY</span>
            <Gauge size={18} />
          </div>
          <h3>Performance with context</h3>
          <p>
            Track resource usage and execution behavior without collapsing every
            trade-off into one headline score.
          </p>
        </article>
        <article className="card amber">
          <div className="card-top">
            <span>CAVEATS</span>
            <CircleAlert size={18} />
          </div>
          <h3>Know what is missing</h3>
          <p>
            Small samples, stochastic outcomes, solver budgets, and hardware
            differences remain visible in the result.
          </p>
        </article>
      </div>

      <section className="text-panel">
        <div className="kicker">Next step</div>
        <h2>
          Bring your runs
          <br />
          <em className="inline-em-eco">into the Studio.</em>
        </h2>
        <p>
          The public surface will expose curated snapshots. Full run exploration
          belongs in the desktop application, where data and configurations can
          be inspected together.
        </p>
        <Link className="btn btn-secondary" to="/studio">
          Explore Studio capabilities <ArrowUpRight size={15} />
        </Link>
      </section>
    </div>
  );
}
