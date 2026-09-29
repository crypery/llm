/*
 * Git: https://github.com/crypery
 * Author: https://crypery.com
 * License: GNU AGPL v3 (Affero GPL)
 */


// tools.h - Windows system tools used by the LLM tool-calling demo
// (system volume control and app launching).
#ifndef TOOLS_H
#define TOOLS_H

#ifdef __cplusplus
extern "C" {
#endif

// system master volume in 0..100 (relative, 100 = max), -1 on error
int tools_volume_get(void);

// set system master volume to v (0..100), 0 on success, -1 on error
int tools_volume_set(int v);

// step system volume by delta (e.g. +10 / -10); returns the new 0..100 or -1
int tools_volume_step(int delta);

// get the mute state: 1 = muted, 0 = audible, -1 on error
int tools_mute_get(void);

// set the mute state (1 = muted / sound off, 0 = unmuted / sound on), 0 on success
int tools_mute_set(int muted);

// launch an application by name (analogous to the Run dialog), 0 on success
int tools_launch(const char * app);

#ifdef __cplusplus
}
#endif

#endif // TOOLS_H