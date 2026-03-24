import { Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar/Navbar';
import RouteFinder from './pages/RouteFinder/RouteFinder';
import ModelComparison from './pages/ModelComparison/ModelComparison';
import TrafficForecast from './pages/TrafficForecast/TrafficForecast';
import Settings from './pages/Settings/Settings';
import './styles/global.css';


export default function App() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
      <Navbar />
      <Routes>
        <Route path="/"           element={<RouteFinder />}     />
        <Route path="/comparison" element={<ModelComparison />} />
        <Route path="/forecast"   element={<TrafficForecast />} />
        <Route path="/settings"   element={<Settings />}        />
        <Route path="*"           element={<Navigate to="/" />} />
      </Routes>
    </div>
  );
}