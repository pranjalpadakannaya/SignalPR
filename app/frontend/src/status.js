import { createContext, useContext } from "react";
export const StatusCtx = createContext({ status: null, refresh: () => {} });
export const useStatus = () => useContext(StatusCtx);
