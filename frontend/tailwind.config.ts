import type { Config } from "tailwindcss";

/**
 * Palette follows the ECDIS night colour table — the scheme a ship's bridge
 * switches to after dark, so instrumentation stays readable without wrecking
 * night vision. Deep blue-black sea, desaturated chart linework, and magenta
 * reserved for hazards, which is where the attributed vessel belongs. Cyan is
 * survey data: the drift cloud. Cyan against magenta is a real chart
 * relationship, not decoration.
 */
const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        sea: {
          abyss: "#050D14",
          deep: "#08151E",
          mid: "#0C1E2A",
          shelf: "#132936",
        },
        chart: {
          line: "#22414F",
          edge: "#2E5464",
          land: "#1A2721",
          coast: "#3C6274",
        },
        ink: {
          bright: "#E6F1F5",
          DEFAULT: "#C2D5DE",
          dim: "#7A929F",
          faint: "#526976",
        },
        hazard: "#E8459B",
        caution: "#F0A93C",
        survey: "#79D2E8",
        slick: "#8C7BD8",
        clear: "#4FBF9B",
      },
      fontFamily: {
        sans: [
          "Segoe UI Variable Text",
          "Segoe UI",
          "ui-sans-serif",
          "system-ui",
          "-apple-system",
          "Helvetica Neue",
          "sans-serif",
        ],
      },
      fontSize: {
        micro: ["10px", { lineHeight: "14px", letterSpacing: "0.01em" }],
        tiny: ["11px", { lineHeight: "15px" }],
        small: ["12px", { lineHeight: "17px" }],
        base: ["13px", { lineHeight: "19px" }],
        readout: ["15px", { lineHeight: "20px", letterSpacing: "-0.01em" }],
        figure: ["27px", { lineHeight: "30px", letterSpacing: "-0.025em" }],
        hero: ["40px", { lineHeight: "40px", letterSpacing: "-0.03em" }],
      },
      borderRadius: {
        chart: "2px",
      },
    },
  },
  plugins: [],
};

export default config;
