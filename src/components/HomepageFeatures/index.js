import clsx from 'clsx';
import Heading from '@theme/Heading';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: 'Sensor Data Processing',
    Svg: require('@site/static/img/sensor.png').default,
    description: (
      <>
        Leverage advanced algorithms to process and interpret raw data from
        LiDAR, cameras, radar, and other sensors for accurate perception.
      </>
    ),
  },
  {
    title: 'Object Fusion and Tracking',
    Svg: require('@site/static/img/nav.png').default,
    description: (
      <>
        Integrate data from multiple sensors to track objects seamlessly,
        ensuring robust environment modeling for real-time applications.
      </>
    ),
  },
  {
    title: 'Vehicle Guidance',
    Svg: require('@site/static/img/gyro.png').default,
    description: (
      <>
        Implement precise trajectory planning and control mechanisms to navigate
        dynamic environments with safety and efficiency.
      </>
    ),
  },
  {
    title: 'Connected Driving',
    Svg: require('@site/static/img/gps.png').default,
    description: (
      <>
        Enable vehicle-to-vehicle (V2V) and vehicle-to-infrastructure (V2I)
        communication to optimize traffic flow and enhance safety.
      </>
    ),
  },
];


function Feature({ Svg, title, description }) {
  return (
    <div className={clsx('col col--6')}>
      <div className="text--center">
        <div
          style={{
            width: '100px', // Adjust the width as needed
            height: '100px', // Adjust the height as needed
            backgroundColor: 'transparent', // Optional: set background color
          }}
        ></div>
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}


/*
function Feature({ Svg, title, description }) {
  return (
    <div className={clsx('col col--6')}>
      <div className="text--center">
        /<img src={Svg} alt={title} className={styles.featureSvg} />
      </div>
      <div className="text--center padding-horiz--md">
        <Heading as="h3">{title}</Heading>
        <p>{description}</p>
      </div>
    </div>
  );
}
*/

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
