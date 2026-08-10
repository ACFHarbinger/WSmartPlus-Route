import { NavLink, Outlet } from "react-router-dom";
import { githubRepo, navLinks } from "../data/site";

export default function Layout() {
  return (
    <div className="site-shell">
      <div className="atmosphere" aria-hidden="true">
        <div className="orb orb-eco" />
        <div className="orb orb-route" />
        <div className="orb orb-amber" />
      </div>

      <header className="site-nav">
        <div className="site-nav-inner">
          <NavLink to="/" className="brand" end>
            <img
              src="/assets/logo-wsmartroute-white.png"
              alt="WSmart+ Route"
              width={120}
              height={36}
            />
            <span className="brand-text">
              <strong>WSmart+ Route</strong>
              <em>routing intelligence</em>
            </span>
          </NavLink>

          <nav className="nav-links" aria-label="Primary">
            {navLinks.map((link) => (
              <NavLink
                key={link.to}
                to={link.to}
                className={({ isActive }) => (isActive ? "active" : undefined)}
              >
                {link.label}
              </NavLink>
            ))}
          </nav>

          <a
            className="nav-cta"
            href={githubRepo}
            target="_blank"
            rel="noreferrer"
          >
            GitHub
          </a>
        </div>
      </header>

      <main className="page">
        <Outlet />
      </main>

      <footer className="site-footer">
        <span>WSmart+ Route · combinatorial optimization for waste collection</span>
        <span>Research platform · Studio desktop · open methods</span>
      </footer>
    </div>
  );
}
