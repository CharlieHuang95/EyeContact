std::string ftos(float f, int nd) {
	std::ostringstream ostr;
	int tens = std::stoi("1" + std::string(nd, '0'));
	ostr << round(f*tens) / tens;
	return ostr.str();
}