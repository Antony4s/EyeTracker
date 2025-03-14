# EyeTracker  

EyeTracker is a **C++ application** that uses **OpenCV** and **dlib** to track eye movements and control the mouse cursor. The program enables **hands-free cursor control** by moving the mouse based on **eye movements** and triggering a **double-click with a blink**.

---

## **Features**  

- **Eye-Based Mouse Control**: Moves the cursor **based on pupil movement**, not head movement.  
- **Blink-to-Click**: **A double blink triggers a double-click** for hands-free interaction.  
- **Real-Time Tracking**: Uses **dlib’s face detector** and **landmark predictor** for accurate eye tracking.  
- **Smooth Mouse Movement**: Reduces jitter and ensures **precise control**.  
- **Customizable Settings**: Adjustable sensitivity for cursor movement and blink detection.  
- **Hands-Free Interaction**: Useful for **accessibility**, **AI research**, and **experimental UI control**.  

---

## **How to Install/Use**  

### **📥 Prerequisites**  
1. **Windows OS** (Tested on Windows 10/11).  
2. **C++ Compiler** (MSVC with Visual Studio).  
3. **[vcpkg](https://vcpkg.io/en/getting-started.html)** (for dependency management).  

### **📌 Installation**  
1. **Clone the repository**:  
   ```sh
   git clone https://github.com/yourusername/EyeTracker.git
   cd EyeTracker

2. **Install dependencies using vcpkg**
   ```sh
   vcpkg install opencv4:x64-windows dlib:x64-windows
3. **Open the project in Visual Studio**
4. **Set build mode to Release and compile**
5. **Run the application**
