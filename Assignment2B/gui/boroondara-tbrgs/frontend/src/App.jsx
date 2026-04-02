import { Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar/Navbar';
import RouteFinder from './pages/RouteFinder/RouteFinder';
import ModelComparison from './pages/ModelComparison/ModelComparison';
import './styles/global.css';


export default function App() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
      <Navbar />
      <Routes>
        <Route path="/"           element={<RouteFinder />}     />
        <Route path="/comparison" element={<ModelComparison />} />
        <Route path="*"           element={<Navigate to="/" />} />
      </Routes>
    </div>
  );
}