import { createBrowserRouter } from "react-router";
import { Root } from "./components/Root";
import { UploadPage } from "./components/UploadPage";
import { ResultsPage } from "./components/ResultsPage";

export const router = createBrowserRouter([
  {
    path: "/",
    Component: Root,
    children: [
      { index: true, Component: UploadPage },
      { path: "results", Component: ResultsPage },
    ],
  },
]);
