import { RouterProvider } from 'react-router';
import { router } from './routes';
import { SegmentationSessionProvider } from './context/SegmentationSessionContext';

export default function App() {
  return (
    <SegmentationSessionProvider>
      <RouterProvider router={router} />
    </SegmentationSessionProvider>
  );
}