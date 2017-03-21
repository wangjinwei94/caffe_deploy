// TODO: Jinwei:
// if this file is included in your project indirectly by some other
// files such as #include <caffe/caffe.hpp> but you want to use glog
// rather than glog_deploy, just #include <glog/logging.h> before
// those files.

#ifndef GLOG_DEPLOY_HPP_
#define GLOG_DEPLOY_HPP_

#include <iostream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <sstream>
#include <ctime>
#include <stdexcept>

#if defined(__linux__)
#include <unistd.h>
#include <sys/time.h>
static inline int __GetTimeUS(void) throw() {
	timeval time;
	gettimeofday(&time, nullptr);
	return time.tv_usec;
}
#else  // defined(__linux__)
static inline int getpid(void) throw() {
	return 0;
}
static inline int __GetTimeUS(void) throw() {
	return 0;
}
#endif  // defined(__linux__)

static std::string __TimeString(void) {
	std::time_t time=std::time(nullptr);
	const std::tm& time_local=*std::localtime(&time);
	static const size_t buf_len=21;
	char buf[buf_len];
	snprintf(buf, buf_len, "%02d%02d %02d:%02d:%02d.%06d", time_local.tm_mon+1, time_local.tm_mday, time_local.tm_hour,
			time_local.tm_min, time_local.tm_sec, __GetTimeUS());
	return std::string(buf);
}

#if defined(__linux__) && defined(__GNUC__)
#include <execinfo.h>
#include <cxxabi.h>
static inline std::string __Demangle(const std::string& name) throw() {
	char* res=abi::__cxa_demangle(&(name[0]), nullptr, nullptr, nullptr);
	if(res==nullptr) {
		return name;
	}
	else {
		std::string res_str(res);
		free(res);
		return res_str;
	}
}
static std::string __StackTrace(void) {
	std::string s;
	s.reserve(1024);
	static const int max_trace_num=20;
	void *buffer[max_trace_num];
	int trace_num=backtrace(buffer, max_trace_num);
	char **trace_string=backtrace_symbols(buffer, trace_num);
	s.append("Call stack trace:\n");
	for(int i=0; i<trace_num; i++) {
		std::string raw_name(trace_string[i]);
		int start=raw_name.find_last_of('(');
		int end=raw_name.find_last_of('+');
		if(start!=std::string::npos && end!=std::string::npos) {
			s.append(raw_name.substr(0, start+1));
			s.append(__Demangle(raw_name.substr(start+1, end-start-1)));
			s.append(raw_name.substr(end, std::string::npos)).append("\n");
		}
		else {
			s.append(raw_name).append("\n");
		}
	}
	free(trace_string);
	return s;
}
#else
static std::string __StackTrace(void) {
	return "Call stack trace is not available.\n";
}
#endif  // defined(__linux__) && defined(__GNUC__)

class __GlogDeoloyNullStream: public std::ostream {
public:
	__GlogDeoloyNullStream() throw(): std::ostream(nullptr) {
	}
	template <typename T>
	inline __GlogDeoloyNullStream& operator<<(T const & x) throw() {
		return *this;
	}
};
static __GlogDeoloyNullStream __glog_deploy_null_stream;

class __GlogDeployLogMessage {
public:
	inline __GlogDeployLogMessage(const std::string& level, const std::string& file, const int line) throw():
			level_(level), line_(line) {
		if(level=="FATAL" || enable) {
			file_=file.substr(file.find_last_of('/')+1, std::string::npos);
			stream_=&buffer_;
		}
		else {
			stream_=&__glog_deploy_null_stream;
		}
	}
	inline ~__GlogDeployLogMessage() throw(std::runtime_error) {
		if(enable || level_=="FATAL") {
			std::string info;
			info.reserve(1024);
			info.append(1, level_[0]).append(__TimeString()).append(" ").append(std::to_string(getpid()));
			info.append(" ").append(file_).append(":").append(std::to_string(line_)).append("]\t");
			info.append(buffer_.str()).append("\n");
			if(level_=="FATAL") {
				info.append(__StackTrace());
			}
			if(enable) {
				std::cerr << info;
			}
			if(level_=="FATAL") {
				throw std::runtime_error(info);
			}
		}
	}
	inline std::ostream& Stream(void) throw() {
		return *stream_;
	}
	static bool enable;

private:
	std::string level_;
	std::string file_;
	int line_;
	std::ostringstream buffer_;
	std::ostream* stream_;
};

// if #include <glog/logging.h> before this file, just use tools in <glog/logging.h>
#ifndef _LOGGING_H_

#define LOG(type) __GlogDeployLogMessage(#type, __FILE__, __LINE__).Stream()
#ifdef DEBUG
#define DLOG(type) LOG(type)
#else
#define DLOG(type) if(true) {} else LOG(type)
#endif

#define CHECK(exp) if(exp) {} else LOG(FATAL) << "Check failed: " << #exp
#ifdef DEBUG
#define DCHECK(exp) CHECK(exp)
#else
#define DCHECK(exp) if(true) {} else CHECK(exp)
#endif

#define CHECK_EQ(x, y) CHECK((x) == (y))
#define CHECK_LT(x, y) CHECK((x) < (y))
#define CHECK_GT(x, y) CHECK((x) > (y))
#define CHECK_LE(x, y) CHECK((x) <= (y))
#define CHECK_GE(x, y) CHECK((x) >= (y))
#define CHECK_NE(x, y) CHECK((x) != (y))
#define CHECK_NOTNULL(x) CHECK((x) != (NULL))

#define DCHECK_EQ(x, y) DCHECK((x) == (y))
#define DCHECK_LT(x, y) DCHECK((x) < (y))
#define DCHECK_GT(x, y) DCHECK((x) > (y))
#define DCHECK_LE(x, y) DCHECK((x) <= (y))
#define DCHECK_GE(x, y) DCHECK((x) >= (y))
#define DCHECK_NE(x, y) DCHECK((x) != (y))
#define DCHECK_NOTNULL(x) DCHECK((x) != (NULL))

#endif  // _LOGGING_H_

#endif  // GLOG_DEPLOY_HPP_
