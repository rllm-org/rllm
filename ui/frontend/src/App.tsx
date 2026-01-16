import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Sidebar } from './components/Sidebar';
import { TrainingRunsList } from './components/TrainingRunsList';
import { TrainingRunDetail } from './components/TrainingRunDetail';

const App: React.FC = () => {
  return (
    <div className="flex w-full h-screen overflow-hidden bg-gray-50">
      <Sidebar />
      <main className="flex-1 overflow-y-auto bg-gray-50">
        <Routes>
          <Route path="/" element={<TrainingRunsList />} />
          <Route path="/runs/:sessionId" element={<TrainingRunDetail />} />
        </Routes>
      </main>
    </div>
  );
};

export default App;
