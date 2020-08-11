// MIT License
// 
// Copyright(c) 2018 Ivan Eichhardt
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
// 
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "camera.hpp"
#include <string_view>
#include <fstream>
#include <iterator>

void common::Camera::LoadParams(const std::string_view filename) {

	std::ifstream fileIn(filename.data());

	if (!fileIn.is_open()) {
		throw std::runtime_error("Failed to open Matches file: " + std::string(filename));
	}

	std::vector<double> params;
	std::copy(
		std::istream_iterator<double>(fileIn),
		std::istream_iterator<double>(),
		std::back_inserter(params));

	if (fileIn.bad()) {
		throw std::runtime_error("Failed to load Matches from file: " + std::string(filename));
	}

	set_params(params);
}
