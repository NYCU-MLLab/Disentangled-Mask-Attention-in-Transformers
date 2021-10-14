import { createTheme, ThemeProvider } from '@material-ui/core';
import { useEffect } from 'react';
import './App.css';
import HomePage from './home/HomePage';

const theme = createTheme({
  palette: {
    primary: {
      main: "#BDBDBD"
    },
    secondary: {
      main: "#F5F5F5"
    },
    success: {
      main: "#C8E6C9"
    },
  }
})

function App() {
  useEffect(() => {
    document.title = "Disentangled Transformer Demo";
  });

  return (
    <ThemeProvider theme={theme}>
      <div className="App">
        <HomePage/>
      </div>
    </ThemeProvider>
  );
}

export default App;
