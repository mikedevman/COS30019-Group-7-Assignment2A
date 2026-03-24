import { ROUTE_COLORS, SCATS_SITES } from '../../utils/constants';
import './RouteCard.css';

function siteName(id) {
  const s = SCATS_SITES.find(x => x.id === id);
  return s ? s.name.split('/')[0].trim() : id;
}

export default function RouteCard({ route, index, selected, onSelect }) {
  const color = ROUTE_COLORS[index % ROUTE_COLORS.length];

  return (
    <button
      className={`route-card${selected ? ' route-card--selected' : ''}${route.best ? ' route-card--best' : ''}`}
      onClick={() => onSelect(route.id === selected ? null : route.id)}
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
        <div className="rc-via">{route.via}</div>
      </div>
    </button>
  );
}
