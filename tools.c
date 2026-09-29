/*
 * Git: https://github.com/crypery
 * Author: https://crypery.com
 * License: GNU AGPL v3 (Affero GPL)
 */


// tools.c - Windows implementation of the tools module (volume + app launch).
// Volume uses the Windows Core Audio API (IAudioEndpointVolume); app launching
// uses ShellExecute (the same lookup the Run dialog performs).
#define COBJMACROS
#include "tools.h"
#include <windows.h>
#include <mmdeviceapi.h>
#include <endpointvolume.h>
#include <ctype.h>

// Core Audio GUIDs (defined explicitly; MinGW does not auto-provide __uuidof)
static const CLSID clsid_mmdev = {0xBCDE0395, 0xE52F, 0x467C, {0x8E,0x3D,0xC4,0x57,0x92,0x91,0x69,0x2E}};
static const IID   iid_mmdev   = {0xA95664D2, 0x9614, 0x4F35, {0xA7,0x46,0xDE,0x8D,0xB6,0x36,0x17,0xE6}};
static const IID   iid_aeb     = {0x5CDF2C82, 0x841E, 0x4546, {0x97,0x22,0x0C,0xF7,0x40,0x78,0x22,0x9A}};

// read the current scalar volume (out) or write one (set); 0 on success
static int vol_rw(float * out, float * set) {
    CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    int ret = -1;
    IMMDeviceEnumerator * pEnum = NULL;
    if (CoCreateInstance(&clsid_mmdev, NULL, CLSCTX_ALL, &iid_mmdev, (void **) &pEnum) == S_OK) {
        IMMDevice * pDev = NULL;
        if (IMMDeviceEnumerator_GetDefaultAudioEndpoint(pEnum, eRender, eConsole, &pDev) == S_OK) {
            IAudioEndpointVolume * pVol = NULL;
            if (IMMDevice_Activate(pDev, &iid_aeb, CLSCTX_ALL, NULL, (void **) &pVol) == S_OK) {
                if (set) { if (IAudioEndpointVolume_SetMasterVolumeLevelScalar(pVol, *set, NULL) == S_OK) ret = 0; }
                else     { if (IAudioEndpointVolume_GetMasterVolumeLevelScalar(pVol, out) == S_OK)      ret = 0; }
                IAudioEndpointVolume_Release(pVol);
            }
            IMMDevice_Release(pDev);
        }
        IMMDeviceEnumerator_Release(pEnum);
    }
    CoUninitialize();
    return ret;
}

int tools_volume_get(void) {
    float v = 0.f;
    if (vol_rw(&v, NULL) != 0) return -1;
    int iv = (int) (v * 100.0f + 0.5f);
    if (iv < 0) iv = 0;
    if (iv > 100) iv = 100;
    return iv;
}

int tools_volume_set(int v) {
    if (v < 0) v = 0;
    if (v > 100) v = 100;
    float f = (float) v / 100.0f;
    return vol_rw(NULL, &f);
}

int tools_volume_step(int delta) {
    int cur = tools_volume_get();
    if (cur < 0) return -1;
    if (tools_volume_set(cur + delta) != 0) return -1;
    return tools_volume_get();
}

int tools_mute_get(void) {
    CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    int ret = -1;
    IMMDeviceEnumerator * pEnum = NULL;
    if (CoCreateInstance(&clsid_mmdev, NULL, CLSCTX_ALL, &iid_mmdev, (void **) &pEnum) == S_OK) {
        IMMDevice * pDev = NULL;
        if (IMMDeviceEnumerator_GetDefaultAudioEndpoint(pEnum, eRender, eConsole, &pDev) == S_OK) {
            IAudioEndpointVolume * pVol = NULL;
            if (IMMDevice_Activate(pDev, &iid_aeb, CLSCTX_ALL, NULL, (void **) &pVol) == S_OK) {
                BOOL m = FALSE;
                if (IAudioEndpointVolume_GetMute(pVol, &m) == S_OK) ret = m ? 1 : 0;
                IAudioEndpointVolume_Release(pVol);
            }
            IMMDevice_Release(pDev);
        }
        IMMDeviceEnumerator_Release(pEnum);
    }
    CoUninitialize();
    return ret;
}

int tools_mute_set(int muted) {
    CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
    int ret = -1;
    IMMDeviceEnumerator * pEnum = NULL;
    if (CoCreateInstance(&clsid_mmdev, NULL, CLSCTX_ALL, &iid_mmdev, (void **) &pEnum) == S_OK) {
        IMMDevice * pDev = NULL;
        if (IMMDeviceEnumerator_GetDefaultAudioEndpoint(pEnum, eRender, eConsole, &pDev) == S_OK) {
            IAudioEndpointVolume * pVol = NULL;
            if (IMMDevice_Activate(pDev, &iid_aeb, CLSCTX_ALL, NULL, (void **) &pVol) == S_OK) {
                if (IAudioEndpointVolume_SetMute(pVol, muted ? TRUE : FALSE, NULL) == S_OK) ret = 0;
                IAudioEndpointVolume_Release(pVol);
            }
            IMMDevice_Release(pDev);
        }
        IMMDeviceEnumerator_Release(pEnum);
    }
    CoUninitialize();
    return ret;
}

int tools_launch(const char * app) {
    if (!app || !*app) return -1;
    // sanitize the app name to avoid shell injection
    for (const char * p = app; *p; p++) {
        unsigned char c = (unsigned char) *p;
        if (!(isalnum(c) || c == ' ' || c == '-' || c == '_' || c == '.' ||
              c == '\\' || c == '/' || c == ':' || c == '(' || c == ')')) {
            return -1;
        }
    }
    HINSTANCE r = ShellExecuteA(NULL, "open", app, NULL, NULL, SW_SHOWNORMAL);
    return ((intptr_t) r > 32) ? 0 : -1;
}