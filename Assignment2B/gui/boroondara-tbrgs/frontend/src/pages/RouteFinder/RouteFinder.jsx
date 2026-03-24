import { useState } from 'react';
import Sidebar from '../../components/Sidebar/Sidebar';
import MapView from '../../components/MapView/MapView';
import RouteCard from '../../components/RouteCard/RouteCard';
import { MOCK_ROUTES } from '../../utils/mockData';
import './RouteFinder.css';

export default function RouteFinder() {
  const [routes,        setRoutes]        = useState([]);
  const [origin,        setOrigin]        = useState('2000');
  const [destination,   setDestination]   = useState('3002');
  const [loading,       setLoading]       = useState(false);
  const [selectedRoute, setSelectedRoute] = useState(null);
  const [error,         setError]         = useState(null);
  const [searched,      setSearched]      = useState(false);

  const handleSearch = async (params) => {
    setLoading(true);
    setError(null);
    setSelectedRoute(null);
    setOrigin(params.origin);
    setDestination(params.destination);
    setSearched(true);
    await new Promise(r => setTimeout(r, 350));
    setRoutes(MOCK_ROUTES.slice(0, params.topK));
    setLoading(false);
  };

  return (
    <div className="rf-layout">
      <Sidebar onSearch={handleSearch} loading={loading} />

      <div className="rf-main">
        {/* Map */}
        <div className="rf-map">
          <MapView
            routes={routes}
            origin={origin}
            destination={destination}
            selectedRoute={selectedRoute}
            onSelectRoute={setSelectedRoute}
          />
          {!searched && (
            <div className="map-hint">
              <div className="hint-box">
                <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
                  <circle cx="10" cy="10" r="8.5" stroke="currentColor" strokeWidth="1.2"/>
                  <path d="M10 6v4M10 13v.5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round"/>
                </svg>
                <p>Select an origin and destination, then click <strong>Find routes</strong></p>
              </div>
            </div>
          )}
        </div>

        {/* Route strip */}
        <div className={`rf-routes${routes.length === 0 ? ' rf-routes--empty' : ''}`}>
          {loading && (
            <div className="routes-loading">
              {[1,2,3,4,5].map(i => <div key={i} className="route-skeleton" />)}
            </div>
          )}
          {!loading && routes.length === 0 && searched && (
            <div className="routes-none">No routes found between these intersections.</div>
          )}
          {!loading && routes.map((r, i) => (
            <RouteCard
              key={r.id}
              route={r}
              index={i}
              selected={selectedRoute}
              onSelect={setSelectedRoute}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
