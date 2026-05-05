import { type RouteConfig, index, route } from "@react-router/dev/routes";

export default [
  index("routes/home.tsx"),
  route("p/:panelId", "routes/panel.tsx"),
  route("p/:panelId/*", "routes/panel.tsx", { id: "panel-deep" }),
] satisfies RouteConfig;
