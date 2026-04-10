import { NavLink } from 'react-router-dom';
import './Navbar.css';

export default function Navbar() {
  return (
    <nav className="navbar">
      {/* App brand and location context */}
      <div className="navbar-brand">
        <span className="brand-pulse" />
        <span className="brand-title" aria-label="Traffic-Based Route Guidance System">
          <span className="brand-title-line">TRAFFIC-BASED</span>
          <span className="brand-title-line">ROUTE GUIDANCE SYSTEM</span>
        </span>
        <span className="brand-area">Boroondara</span>
      </div>
      {/* Right-side status indicator */}
      <div className="navbar-right">
        <div className="live-badge">
          <span className="live-dot" />
          <span>Live</span>
        </div>
      </div>
    </nav>
  );
}
