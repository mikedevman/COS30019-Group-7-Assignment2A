// docker run -t -p 5000:5000 -v "C:/Users/Admin/Documents/COS30019-Group-7-Assignment2/Assignment2B/gui/boroondara-tbrgs/backend/osrm:/data" osrm/osrm-backend osrm-routed --algorithm mld /data/australia-260401.osrm

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
  const s = SCATS_SITES.find(x => x.id === id);
  return s ? [s.lat, s.lng] : null;
}

async function fetchRoadPath(coords) {
  const coordStr = coords.map(([lat, lng]) => `${lng},${lat}`).join(';');
  const res = await fetch(
    `http://localhost:5000/route/v1/driving/${coordStr}?overview=full&geometries=geojson`
  );
  const data = await res.json();
  if (!data.routes || !data.routes[0]) return coords;
  return data.routes[0].geometry.coordinates.map(([lng, lat]) => [lat, lng]);
}

export default function MapView({ routes = [], origin, destination, selectedRoute, onSelectRoute }) {
  const [activeMarker, setActiveMarker] = useState(null);
  const [roadPaths, setRoadPaths] = useState([]);

  const tileUrl = OSM_TILE_URL.includes('{apikey}')
    ? OSM_TILE_URL.replace('{apikey}', encodeURIComponent(OSM_API_KEY))
    : OSM_TILE_URL;

  useEffect(() => {
    if (!routes.length) {
      setRoadPaths([]);
      return;
    }
    Promise.all(
      routes.map(r =>
        fetchRoadPath(r.path.map(id => siteLatlng(id)).filter(Boolean))
      )
    ).then(setRoadPaths);
  }, [routes]);

  return (
    <div className="map-wrap">
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
          const isSelected = selectedRoute === null || selectedRoute === route.id;

          return (
            <Polyline
              key={route.id}
              positions={path}
              pathOptions={{
                color: ROUTE_COLORS[i % ROUTE_COLORS.length],
                opacity: isSelected ? 0.85 : 0.25,
                weight: selectedRoute === route.id ? 5 : 3,
              }}
              eventHandlers={{
                click: () => onSelectRoute(route.id === selectedRoute ? null : route.id),
              }}
            />
          );
        })}

        {/* Markers */}
        {SCATS_SITES.map(site => {
          const isOrigin = site.id === origin;
          const isDest = site.id === destination;

          if (!isOrigin && !isDest && !routes.some(r => r.path.includes(site.id))) return null;

          return (
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

