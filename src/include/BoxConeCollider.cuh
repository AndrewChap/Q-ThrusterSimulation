// Checks if our box intersects with our cone.  It's used so that we don't create boxes that can't have particles in them.  Relies on the following:
//		- The box must be cartesian-aligned
//		- The cone must be aligned with z-axis
//		- the bottom of the cone must be at z=0
//		- the top of the cone is defined as z = L
//		- the bottom of the cone has radius R1 (larger radius)
//		- the top of the cone has radius R2 (smaller radius)

float LineCircleCollider(float x1, float x2, float y);

int BoxConeCollider(float px, float py, float pz, float dxB, float dyB, float dzB, float R1, float R2, float L){

	float rbound;

	if (pz - 0.5*dzB >= 0 && pz - 0.5*dzB < L){				// if the bottom of our box is above the bottom of the cone AND if the bottom of our box is below the top of the cone
		rbound = (pz - 0.5*dzB)*(R2 - R1) / L + R1;			// the radius of the check-circle is taken where it will be greatest: at the z position of bottom of the box
	}
	else if (pz - 0.5*dzB <= 0 && pz + 0.5*dzB > 0){		// if the bottom of the box is below the bottom of the cone and the top of the box is above the bottom of the cone
		rbound = R1;										// radius of the check-circle is just the bottom of the cone (big radius)
	}
	else{													// if the bottom of the box is above the top of the cone or the top of the box is below the bottom of the cone then the box can't possibly be intersecting the cone
		return 0;
	}

	// check four z-parallel sides of the box to see if they intersect with rbound
	if (LineCircleCollider(px - 0.5*dxB, px + 0.5*dxB, py - 0.5*dyB) < rbound*rbound){
		return 1;
	}
	if (LineCircleCollider(px - 0.5*dxB, px + 0.5*dxB, py + 0.5*dyB) < rbound*rbound){
		return 1;
	}
	if (LineCircleCollider(py - 0.5*dyB, py + 0.5*dyB, px - 0.5*dxB) < rbound*rbound){
		return 1;
	}
	if (LineCircleCollider(py - 0.5*dyB, py + 0.5*dyB, px + 0.5*dxB) < rbound*rbound){
		return 1;
	}
	else{
		return 0;
	}
}

float LineCircleCollider(float x1, float x2, float y){
	// NOTE: x1 MUST BE LESS THAN x2 FOR THIS TO WORK PROPERLY!
	// Finds the closest point to the origin of the line segment from [x1,y] to [x2,1] and returns the square of the distance from that point to the origin
	if (x1 <= 0 && x2 >= 0){
		return y*y;
	}
	else if (x1 > 0) {
		return x1*x1 + y*y;
	}
	else if (x2 < 0) {
		return x2*x2 + y*y;
	}
}
