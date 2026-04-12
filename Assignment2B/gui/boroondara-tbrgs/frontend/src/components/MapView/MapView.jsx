import { MapContainer, TileLayer, Polyline, CircleMarker, Popup } from 'react-leaflet';
import { useState, useEffect } from 'react';
import {
  MAP_CENTER,
  MAP_ZOOM,
  OSM_API_KEY,
  OSM_ATTRIBUTION,
  OSM_TILE_URL,
  SCATS_SITES,
  ROUTE_COLORS,
} from '../../utils/constants';
import 'leaflet/dist/leaflet.css';
import './MapView.css';

function siteLatlng(id) {
  // Look up latitude/longitude for a SCATS site id
  const s = SCATS_SITES.find(x => x.id === id);
  return s ? [s.lat, s.lng] : null;
}

async function fetchRoadPath(coords) {
  // Fetch snapped road geometry for a sequence of waypoints via OSRM
  const coordStr = coords.map(([lat, lng]) => `${lng},${lat}`).join(';');
  
  let res;

  try {
    // Try public OSRM first for routing geometry
    res = await fetch(
      `http://router.project-osrm.org/route/v1/driving/${coordStr}?overview=full&geometries=geojson`
    );

    if (!res.ok) throw new Error("Public OSRM failed");
  } catch (err) {
    // Fall back to local OSRM instance if public endpoint fails
    console.log("Falling back to ");
  
    res = await fetch(
      `http://localhost:5000/route/v1/driving/${coordStr}?overview=full&geometries=geojson`
    );
  }

  // Convert OSRM GeoJSON coords into Leaflet-friendly [lat, lng] pairs
  const data = await res.json();
  if (!data.routes || !data.routes[0]) return coords;
  return data.routes[0].geometry.coordinates.map(([lng, lat]) => [lat, lng]);
}

export default function MapView({ routes = [], origin, destination, selectedRoute, onSelectRoute, loading = false }) {
  // Track marker selection and resolved road geometries for each route
  const [activeMarker, setActiveMarker] = useState(null);
  const [roadPaths, setRoadPaths] = useState([]);

  // Build tile URL with API key substitution if required
  const tileUrl = OSM_TILE_URL.includes('{apikey}')
    ? OSM_TILE_URL.replace('{apikey}', encodeURIComponent(OSM_API_KEY))
    : OSM_TILE_URL;

  useEffect(() => {
    // Recompute road geometry paths whenever the route list changes
    if (!routes.length || loading) {
      setRoadPaths([]);
      return;
    }
    Promise.all(
      routes.map(r =>
        fetchRoadPath(r.path.map(id => siteLatlng(id)).filter(Boolean))
      )
    ).then(setRoadPaths);
  }, [routes, loading]);

  return (
    <div className="map-wrap">
      {/* Main Leaflet map container */}
      <MapContainer
        center={[MAP_CENTER.lat, MAP_CENTER.lng]}
        zoom={MAP_ZOOM}
        className="gmap"
        style={{ width: '100%', height: '100%' }}
        scrollWheelZoom
      >
        {/* OSM tiles */}
        <TileLayer
          attribution={OSM_ATTRIBUTION}
          url={tileUrl}
        />

        {/* Routes */}
        {roadPaths.map((path, i) => {
          const route = routes[i];
          if (!route) return null;
          const isSelected = selectedRoute === null || selectedRoute === route.id;

          return (
            // Route polyline with click-to-select behaviour
            <Polyline
              key={route.id}
              positions={path}
              pathOptions={{
                color: ROUTE_COLORS[i % ROUTE_COLORS.length],
                opacity: isSelected ? 0.99 : 0.1,
                weight: selectedRoute === route.id ? 5 : 3,
              }}
              eventHandlers={{
                click: () => onSelectRoute(route.id === selectedRoute ? null : route.id),
              }}
            />
          );
        })}

        {/* Markers */}
        {!loading && SCATS_SITES.map(site => {
          const isOrigin = site.id === origin;
          const isDest = site.id === destination;

          // Only render markers that are origin/destination or appear in any displayed route
          if (!isOrigin && !isDest && !routes.some(r => r.path.includes(site.id))) return null;

          return (
            // Marker for each relevant SCATS site with popup on click
            <CircleMarker
              key={site.id}
              center={[site.lat, site.lng]}
              radius={isOrigin || isDest ? 8 : 5}
              pathOptions={{
                color: isOrigin ? '#0F6E56' : isDest ? '#993C1D' : '#888780',
                fillColor: isOrigin ? '#1D9E75' : isDest ? '#D85A30' : '#FFFFFF',
                fillOpacity: 1,
                weight: 2,
              }}
              eventHandlers={{
                click: e => {
                  // Toggle popup open/closed while tracking the active marker id
                  setActiveMarker(prev => {
                    const next = prev === site.id ? null : site.id;
                    if (next === null) e?.target?.closePopup?.();
                    else e?.target?.openPopup?.();
                    return next;
                  });
                },
              }}
            >
              <Popup>
                <div>
                  <p>SCATS {site.id}</p>
                  <p>{site.name}</p>
                </div>
              </Popup>
            </CircleMarker>
          );
        })}
      </MapContainer>

      {/* Legend */}
      {routes.length > 0 && (
        <div className="map-legend">
          {routes.map((r, i) => (
            // Legend row that mirrors route colors and supports selection
            <button
              key={r.id}
              className={`legend-row${selectedRoute === r.id ? ' legend-row--active' : ''}`}
              onClick={() => onSelectRoute(r.id === selectedRoute ? null : r.id)}
            >
              <span className="legend-swatch" style={{ background: ROUTE_COLORS[i % ROUTE_COLORS.length] }} />
              <span className="legend-label">Route {r.id}</span>
              <span className="legend-time">{r.duration} min</span>
              {r.best && <span className="legend-best">Best</span>}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

