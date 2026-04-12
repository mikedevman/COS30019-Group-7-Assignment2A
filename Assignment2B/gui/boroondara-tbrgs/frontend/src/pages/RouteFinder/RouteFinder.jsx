import { useState } from 'react';
import Sidebar from '../../components/Sidebar/Sidebar';
import MapView from '../../components/MapView/MapView';
import RouteCard from '../../components/RouteCard/RouteCard';
import { DEFAULTS } from '../../utils/constants';
import './RouteFinder.css';

export default function RouteFinder() {
  // Page state for current routes, selections, and request lifecycle
  const [routes,        setRoutes]        = useState([]);
  const [origin,        setOrigin]        = useState(DEFAULTS.origin);
  const [destination,   setDestination]   = useState(DEFAULTS.destination);
  const [loading,       setLoading]       = useState(false);
  const [selectedRoute, setSelectedRoute] = useState(null);
  const [error,         setError]         = useState(null);
  const [searched,      setSearched]      = useState(false);

  const handleSearch = async (params) => {
    // Submit a route-search request to the backend and map response into UI-friendly objects
    setLoading(true);
    setError(null);
    setSelectedRoute(null);
    setOrigin(params.origin);
    setDestination(params.destination);
    setSearched(true);
    
    try {
      // Call backend API with selected journey, model, and parameter inputs
      const res = await fetch('http://127.0.0.1:5001/api/route', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          origin: params.origin,
          destination: params.destination,
          model: params.model || 'lstm',
          topK: params.topK || 5,
          speedLimit: params.speedLimit || 60,
          intersectionDelay: params.intersectionDelay ?? 30,
          departTime: params.departTime || '08:30',
          bidirectional: params.bidirectional || false,
        })
      });

      // Convert non-2xx responses into a user-visible error
      if (!res.ok) throw new Error('Backend failed to respond');

      // Parse backend payload and surface backend-side error field
      const data = await res.json();
      if (data.error) throw new Error(data.error);

      // Determine divergence point from best path to compute a simple "via" label
      const bestPath = data.routes[0]?.path || [];
      const parsedRoutes = data.routes.map((r, i) => {
        let diffNode = r.path[1];
        if (i > 0) {
          for (let j = 1; j < r.path.length; j++) {
            if (r.path[j] !== bestPath[j]) {
              diffNode = r.path[j];
              break;
            }
          }
        }
        
        // Normalise backend route shape into components used by MapView and RouteCard
        return {
          id: r.route.toString(),
          best: i === 0,
          duration: r.estimated_time_mins,
          distance: r.distance_km ? `${r.distance_km} km` : 'N/A',
          intersections: Math.max(0, r.path.length - 2),
          via: `Node ${diffNode || 'Unknown'}`,
          path: r.path
        };
      });

      // Update route list and default-select the first (fastest) route
      setRoutes(parsedRoutes);
      if (parsedRoutes.length > 0) setSelectedRoute(parsedRoutes[0].id);

    } catch (err) {
      // Map fetch errors into a readable message for the routes strip
      console.error(err);

      let message = "Failed to fetch routes.";

      if (err.message === "Failed to fetch") {
        message += " Cannot connect to backend.";
      } else {
        message += " " + err.message;
      }

      setError(message);
    } finally {
      // Always clear loading flag once request completes
      setLoading(false);
    }
  };

  return (
    // Two-column layout: sidebar controls + main map and route strip
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
            loading={loading}
          />
          {!searched && (
            // Onboarding hint shown until the first search is performed
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
            // Loading skeletons while backend request is in flight
            <div className="routes-loading">
              {[1,2,3,4,5].map(i => <div key={i} className="route-skeleton" />)}
            </div>
          )}

          {!loading && routes.length === 0 && searched && !error && (
            // Empty-state message when search succeeds but no routes are returned
            <div className="routes-none">No routes found between these intersections.</div>
          )}

          {!loading && routes.length === 0 && searched && error && (
            // Error message when backend request fails or returns an error
            <div className="routes-none">{error}</div>
          )}

          {!loading && routes.map((r, i) => (
            // Route cards allow selecting a specific route for emphasis on the map
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