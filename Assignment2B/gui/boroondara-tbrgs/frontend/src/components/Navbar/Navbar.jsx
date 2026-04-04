import { NavLink } from 'react-router-dom';
import './Navbar.css';

export default function Navbar() {
  return (
    <nav className="navbar">
      <div className="navbar-brand">
        <span className="brand-pulse" />
        <span className="brand-name">TBRGS</span>
        <span className="brand-area">Boroondara</span>
      </div>
      <div className="navbar-right">
        <div className="live-badge">
          <span className="live-dot" />
          <span>Live</span>
        </div>
      </div>
    </nav>
  );
}
