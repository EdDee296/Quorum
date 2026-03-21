import { Outlet } from 'react-router';
import { Microscope } from 'lucide-react';

export function Root() {
  return (
    <div className="min-h-screen" style={{ backgroundColor: '#F5F5F5' }}>
      {/* Header */}
      <header style={{ backgroundColor: '#304C64' }} className="shadow-sm">
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
          <div className="flex flex-col items-center gap-3">
            <Microscope className="h-8 w-8" style={{ color: '#A4CCD4' }} />
            <div className="text-center">
              <h1 style={{ color: 'white' }}>Sub-Cellular Instance Segmentation</h1>
              <p className="text-sm mt-1" style={{ color: '#A4CCD4' }}>
                Automated segmentation of chromocenter, nuclei, and background
              </p>
            </div>
          </div>
        </div>
      </header>

      <Outlet />

      {/* Footer */}
      <footer className="mt-16 border-t" style={{ backgroundColor: 'white', borderColor: '#A4CCD4' }}>
        <div className="max-w-7xl mx-auto px-4 py-6 sm:px-6 lg:px-8">
        </div>
      </footer>
    </div>
  );
}
