import {AppBar, Toolbar, InputLabel, Typography, MenuItem, FormHelperText, Select, FormControl, createTheme, ThemeProvider } from '@material-ui/core';
import { makeStyles } from '@material-ui/core';

const theme = createTheme({
    palette: {
        primary: {
            main: '#F5F5F5'
        }
    }
});

const useStyles = makeStyles((theme) => ({
    root: {
        width: '100%',
        backgroundColor: '#F5F5F5'
    },
    menuButton: {
        marginRight: theme.spacing(2)
    },
    title: {
        flexGrow: 1
    },
    formControl: {
        margin: theme.spacing(1),
        minWidth: 300
    },
    selectEmpty: {
        marginTop: theme.spacing(1)
    }
}));

function HomeNavBar(props) {
    const classes = useStyles();

    return (
        <div className={classes.root}>
            <ThemeProvider theme={theme}>
                <AppBar color="primary" position="fixed">
                    <Toolbar>
                        <Typography variant="h5" className={classes.title} align="center">
                        Disentangled Mask Attention in Transformers
                        </Typography>
                        <FormControl className={classes.formControl}>
                            <InputLabel>Task</InputLabel>
                            <Select value={props.task} onChange={props.onTaskChange}>
                                <MenuItem value="MT">Machine Translation (MT)</MenuItem>
                                <MenuItem value="ASR">Automatic speech recognition (ASR)</MenuItem>
                            </Select>
                            <FormHelperText>Please select one task.</FormHelperText>
                        </FormControl>
                    </Toolbar>
                </AppBar>
            </ThemeProvider>
        </div>
    );
}

export default HomeNavBar