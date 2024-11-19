# Automated and Connected Driving Challenges (ACDC)

Welcome to the **Automated and Connected Driving Challenges (ACDC)** repository. This repository complements the [ACDC - MOOC](https://learning.edx.org/course/course-v1:RWTHx+ACDC+3T2023/home) offered by RWTH Aachen University on edX. The course provides insights into the development and testing of automated and connected driving functions, guiding participants through the prototypical development of an automated vehicle.

---

## Course Overview

The ACDC course introduces participants to the latest research challenges in automated and connected driving. It offers a step-by-step approach to developing and testing driving functions, covering topics such as sensor data processing, environment modeling, trajectory planning, and vehicle guidance. The course utilizes the Robot Operating System (ROS) for implementing software modules.

---

## Repository Contents

This repository contains Jupyter Notebook programming tasks designed to reinforce the concepts covered in the ACDC course. The notebooks are organized into sections corresponding to the course modules:

- **Section 1: Introduction & Robot Operating System**
- **Section 2: Sensor Data Processing**
- **Section 3: Object Fusion and Tracking**
- **Section 4: Vehicle Guidance**
- **Section 5: Connected Driving**

Each section includes exercises and examples to facilitate hands-on learning.

---

## mdBook Documentation

We provide an **mdBook** for structured, user-friendly documentation to enhance your learning experience.

### How to Use the mdBook

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/ika-rwth-aachen/acdc-notebooks.git
   ```

2. **Navigate to the `mdBook` Directory:**

   ```bash
   cd acdc-notebooks/mdbook
   ```

3. **Install mdBook:**

   Ensure you have Rust installed. If not, install Rust using [rustup](https://rustup.rs/).

   ```bash
   cargo install mdbook
   ```

4. **Serve the Documentation Locally:**

   Run the following command to serve the book locally:

   ```bash
   mdbook serve
   ```

   The documentation will be accessible at `http://localhost:3000`.

5. **Build the Documentation (Optional):**

   To generate the static files for the documentation:

   ```bash
   mdbook build
   ```

   The output will be available in the `book` directory.

---

## Getting Started

To get started with the notebooks:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/ika-rwth-aachen/acdc-notebooks.git
   ```

2. **Navigate to the Directory:**

   ```bash
   cd acdc-notebooks
   ```

3. **Set Up the Environment:**

   Ensure you have Python and Jupyter Notebook installed. You can create a virtual environment and install the required dependencies using:

   ```bash
   python3 -m venv acdc-env
   source acdc-env/bin/activate
   pip install -r requirements.txt
   ```

4. **Launch Jupyter Notebook:**

   ```bash
   jupyter notebook
   ```

5. **Access the Notebooks:**

   In the Jupyter interface, navigate to the desired section and open the corresponding notebook to begin.

---

## Additional Resources

- **Course Wiki:** For detailed information and additional resources, visit the [ACDC Wiki](https://github.com/ika-rwth-aachen/acdc/wiki).

- **edX Course Page:** Enroll in the course and access all materials on the [edX platform](https://learning.edx.org/course/course-v1:RWTHx+ACDC+3T2023/home).

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This course and repository are developed by the Institute for Automotive Engineering (ika) at RWTH Aachen University. We thank all contributors and participants for their support and engagement.
