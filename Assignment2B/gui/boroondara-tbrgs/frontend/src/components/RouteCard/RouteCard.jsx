import { useState, useRef } from 'react';
import { createPortal } from 'react-dom';
import { ROUTE_COLORS, SCATS_SITES } from '../../utils/constants';
import './RouteCard.css';

function siteName(id) {
  const s = SCATS_SITES.find(x => x.id === id);
  return s ? s.name : id;
}

export default function RouteCard({ route, index, selected, onSelect }) {
  const color = ROUTE_COLORS[index % ROUTE_COLORS.length];
  const nodes = Array.isArray(route?.path) ? route.path : [];
  
  const [isHovered, setIsHovered] = useState(false);
  const [pos, setPos] = useState({ top: 0, left: 0, width: 0 });
  const cardRef = useRef(null);

  const handleMouseEnter = () => {
    if (cardRef.current) {
      const rect = cardRef.current.getBoundingClientRect();
      setPos({ top: rect.top, left: rect.left, width: rect.width });
    }
    setIsHovered(true);
  };

  return (
    <button
      ref={cardRef}
      className={`route-card${selected ? ' route-card--selected' : ''}${route.best ? ' route-card--best' : ''}`}
      onClick={() => onSelect(route.id === selected ? null : route.id)}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={() => setIsHovered(false)}
      style={{ '--rc': color }}
    >
      <div className="rc-stripe" />
      <div className="rc-body">
        <div className="rc-top">
          <span className="rc-num">Route {route.id}</span>
          {route.best && <span className="rc-badge">Fastest</span>}
        </div>
        <div className="rc-time">{route.duration}<span className="rc-unit"> min</span></div>
        <div className="rc-meta">
          <span>{route.distance} km</span>
          <span className="rc-dot" />
          <span>{route.intersections} stops</span>
        </div>
      </div>

      {isHovered && nodes.length > 0 && createPortal(
        <div 
          className="rc-hover-popup" 
          role="tooltip" 
          aria-hidden="true"
          style={{
            position: 'fixed',
            left: `${pos.left + 8}px`,
            width: `${pos.width - 16}px`,
            bottom: `${window.innerHeight - pos.top + 8}px`,
            opacity: 1,
            pointerEvents: 'none',
            transform: 'translateY(0)'
          }}
        >
          <div className="rc-hover-title">Sites in path</div>
          <ol className="rc-hover-list">
            {nodes.map((id, i) => (
              <li key={`${route.id}-${id}-${i}`} className="rc-hover-item">
                <span className="rc-hover-id">{id}</span>
                <span className="rc-hover-name">{siteName(id)}</span>
              </li>
            ))}
          </ol>
        </div>,
        document.body
      )}
    </button>
  );
}
