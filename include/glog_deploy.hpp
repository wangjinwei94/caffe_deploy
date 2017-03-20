// TODO: Jinwei:
// if this file is included in your project indirectly by some other
// files such as #include <caffe/caffe.hpp> but you want to use glog
// rather than glog_deploy, just #include <glog/logging.h> before
// those files.

#ifndef GLOG_DEPLOY_HPP_
#define GLOG_DEPLOY_HPP_

#include <iostream>
#include <string>
#include <cstdlib>
#include <sstream>
#include <ctime>

#if defined(__linux__)
#include <unistd.h>
#include <sys/time.h>
static inline int gettimeus(void) {
	timeval time;
	gettimeofday(&time, nullptr);
	return time.tv_usec;
}
#else  // defined(__linux__)
static inline int getpid(void) {
	return 0;
}
static inline int gettimeus(void) {
	return 0;
}
#endif  // defined(__linux__)

#if defined(__linux__) && defined(__GNUC__)
#include <execinfo.h>
#include <cxxabi.h>
static inline std::string __Demangle(const std::string& name) {
	size_t size;
	int status;
	char* res=abi::__cxa_demangle(&(name[0]), nullptr, &size, &status);
	if(res==nullptr) {
		return name;
	}
	else {
		std::string res_str(res);
		free(res);
		return res_str;
	}
}
#define GLOG_DEPLOY_HAS_BACKTRACE
#endif  // defined(__linux__) && defined(__GNUC__)

class __GlogDeoloyNullStream: public std::ostream {
public:
	inline __GlogDeoloyNullStream(): std::ostream(nullptr) {
	}
	template <typename T>
	inline __GlogDeoloyNullStream& operator<<(T const & x) {
		return *this;
	}
};
static __GlogDeoloyNullStream __glog_deploy_null_stream;

class __GlogDeployLogMessage {
public:
	inline __GlogDeployLogMessage(const std::string& level, const std::string& file, const int line):
			level_(level), line_(line) {
		if(level=="FATAL" || enable) {
			file_=file.substr(file.find_last_of('/')+1, std::string::npos);
			stream_=&buffer_;
		}
		else {
			stream_=&__glog_deploy_null_stream;
		}
	}
	inline ~__GlogDeployLogMessage() throw() {
		if(!enable && level_!="FATAL") {
			return;
		}

		Stream() << std::endl;
		if(level_=="FATAL") {
#ifdef GLOG_DEPLOY_HAS_BACKTRACE
			const int max_trace_num=20;
			void *buffer[max_trace_num];
			int trace_num=backtrace(buffer, max_trace_num);
			char **trace_string=backtrace_symbols(buffer, trace_num);
			Stream() << "Call stack trace:" << std::endl;
			for(int i=0; i<trace_num; i++) {
				std::string raw_name(trace_string[i]);
				int start=raw_name.find_last_of('(');
				int end=raw_name.find_last_of('+');
				if(start!=std::string::npos && end!=std::string::npos) {
					Stream() << raw_name.substr(0, start+1);
					Stream() << __Demangle(raw_name.substr(start+1, end-start-1));
					Stream() << raw_name.substr(end, std::string::npos) << std::endl;
				}
				else {
					Stream() << raw_name << std::endl;
				}
			}
			free(trace_string);
#else  // GLOG_DEPLOY_HAS_BACKTRACE
			Stream() <<"Call stack trace is not available." << std::endl;
#endif  // GLOG_DEPLOY_HAS_BACKTRACE
		}
		std::string info=InfoString();
		if(enable) {
			std::cerr << info << buffer_.str();
		}
		if(level_=="FATAL") {
			throw std::runtime_error(info+buffer_.str());
		}
	}
	inline std::ostream& Stream(void) {
		return *stream_;
	}
	static bool enable;

private:
	inline std::string InfoString(void) {
		std::time_t time=std::time(nullptr);
		const std::tm& time_local=*std::localtime(&time);
		char buf[1000];
		sprintf(buf, "%02d%02d %02d:%02d:%02d.%06d", time_local.tm_mon+1, time_local.tm_mday, time_local.tm_hour,
				time_local.tm_min, time_local.tm_sec, gettimeus());
		std::ostringstream ss;
		ss << level_[0] << buf << " " << getpid() << " " << file_ << ":" << line_ << "]\t";
		return ss.str();
	}
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
